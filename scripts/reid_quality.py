"""
reid_quality.py — how well does the DEVELOPED re-ID (contrastive head) recognise INDIVIDUAL fish?

Evaluated on a stretch where identity is TRUSTED (one clean no-swap window), so labels are ground truth.
Produces a 3-panel report:
  1. HOLD-TIME curve  — kNN identity accuracy vs the time gap between query & match (does ID survive as the
                        fish ages? raw frozen vs contrastive). This is the "retain after N seconds" answer.
  2. PER-FISH accuracy — which individuals it nails vs which are hard (the look-alikes).
  3. CONFUSION matrix — when it's wrong, WHICH fish does it mistake for which (does it cluster the same ones?).

NOTE ON 30s: the longest TRUSTED window is ~20s (your longest clean no-swap curation window), so the curve
is measured to ~18s. Beyond that we have no swap-free ground truth on single-session data — the trend
(declining) is the honest extrapolation. Measuring true 30s+ retention needs a longer verified window.

Usage:  python -m scripts.reid_quality --video_name IMG_1839 --stretch 04
"""
import os
import argparse
import logging

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reid_features import build_features
from scripts.console import banner
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)
FPS = 59.78


def knn_labels(emb):
    """Leave-one-out nearest-neighbour index for every crop (cosine)."""
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-2.0)
    return sim, sim.argmax(dim=1)


def main(video_name, stretch):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    run_dir = full["finetune_reid"]["videos"][video_name]["crops_run"]
    backbone = full["finetune_reid"]["videos"][video_name]["backbone"]

    banner(f"RE-ID QUALITY — contrastive head on trusted stretch {stretch} ({video_name})")
    feats, labels, frames, label_map = build_features(run_dir, [stretch], backbone)
    slot_to_fish = {s: f for f, s in label_map.items()}
    lab = labels.numpy(); fr = frames.numpy().astype(float)
    span = (fr.max() - fr.min()) / FPS
    logger.info(f"{len(feats)} crops, {len(label_map)} fish, {span:.1f}s trusted span")

    # contrastive projection
    ck = torch.load(os.path.join(run_dir, "stitch", "contrastive_head.pt"), map_location="cpu")
    head = torch.nn.Sequential(torch.nn.Linear(ck["in_dim"], ck["hidden_dim"]), torch.nn.ReLU(),
                               torch.nn.Linear(ck["hidden_dim"], ck["out_dim"]))
    head.load_state_dict({k.replace("net.", ""): v for k, v in ck["head_state"].items()}); head.eval()
    with torch.no_grad():
        con = head(feats)

    fish = sorted(set(lab.tolist()))                           # slot ids present (used by all 3 panels)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))

    # ── panel 1: ID retention per fish + RELIABILITY HORIZON (where the "DNA" stops being trustworthy) ──
    RELIABLE = 0.90                                             # accuracy floor we call "trustworthy"
    secs = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18]
    e = F.normalize(con, dim=1); simc = (e @ e.T).numpy(); gap = np.abs(fr[:, None] - fr[None, :])

    def acc_at(qmask, s):
        S = simc.copy(); S[gap < s * FPS] = -2
        ok = tot = 0
        for i in np.where(qmask)[0]:
            j = S[i].argmax()
            if S[i, j] < -1:
                continue
            ok += int(lab[j] == lab[i]); tot += 1
        return ok / tot if tot else np.nan

    def horizon(curve):                                        # last gap where accuracy is still >= RELIABLE
        return max([s for s, a in zip(secs, curve) if a is not np.nan and a >= RELIABLE], default=0)

    overall = [acc_at(np.ones(len(lab), bool), s) for s in secs]
    per_fish_h = {}
    for si, sl in enumerate(fish):
        pf = [acc_at(lab == sl, s) for s in secs]
        per_fish_h[slot_to_fish[sl]] = horizon(pf)
        ax[0].plot(secs, pf, "-", lw=1.2, alpha=.55, color=plt.cm.tab10(si), label=f"fish{slot_to_fish[sl]}")
    ax[0].plot(secs, overall, "k-o", lw=2.5, label="all fish")
    H = horizon(overall)
    ax[0].axhline(RELIABLE, ls="--", c="green", alpha=.7, label=f"{int(RELIABLE*100)}% reliable")
    ax[0].axvline(H, ls=":", c="red", lw=2)
    ax[0].text(H + 0.3, 0.25, f"HORIZON\n~{H}s", color="red", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("time gap query↔match (s)"); ax[0].set_ylabel("identity accuracy"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title(f"reliability horizon: fingerprint trustworthy to ~{H}s"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=.3)

    # ── panel 2: per-fish accuracy (contrastive) ──
    _, nn = knn_labels(con)
    nn = nn.numpy(); hit = (lab[nn] == lab)
    fish = sorted(set(lab.tolist())); accs = [hit[lab == s].mean() for s in fish]
    ax[1].bar([f"fish{slot_to_fish[s]}" for s in fish], accs, color=plt.cm.tab10(range(len(fish))))
    ax[1].axhline(1, ls=":", c="grey"); ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("kNN accuracy")
    ax[1].set_title("per-fish recognition (which individuals are hard?)")
    for i, a in enumerate(accs):
        ax[1].text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)

    # ── panel 3: confusion matrix (contrastive) ──
    n = len(fish); conf = np.zeros((n, n))
    for i in range(len(lab)):
        conf[lab[i], lab[nn[i]]] += 1
    conf = conf / conf.sum(1, keepdims=True)
    im = ax[2].imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax[2].set_xticks(range(n)); ax[2].set_yticks(range(n))
    ax[2].set_xticklabels([f"fish{slot_to_fish[s]}" for s in fish]); ax[2].set_yticklabels([f"fish{slot_to_fish[s]}" for s in fish])
    ax[2].set_xlabel("recognised as"); ax[2].set_ylabel("true fish")
    ax[2].set_title("confusion (who gets mistaken for whom)")
    for i in range(n):
        for j in range(n):
            if conf[i, j] > 0.02:
                ax[2].text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                           color="white" if conf[i, j] > 0.5 else "black", fontsize=8)

    fig.suptitle(f"Developed re-ID quality — contrastive head, stretch {stretch} (trusted labels, {span:.0f}s span)")
    fig.tight_layout()
    out = os.path.join(run_dir, "stitch", f"reid_quality_stretch{stretch}.png")
    fig.savefig(out, dpi=130); plt.close(fig)

    banner("SUMMARY")
    logger.info(f"overall contrastive kNN accuracy: {hit.mean():.3f}")
    logger.info(f"RELIABILITY HORIZON (fingerprint stays >={int(RELIABLE*100)}% accurate): ~{H}s overall")
    logger.info("per-fish horizon (when each individual's 'DNA' stops being trustworthy):")
    for f_id in sorted(per_fish_h):
        logger.info(f"  fish{f_id}: reliable up to ~{per_fish_h[f_id]}s")
    worst = min(per_fish_h, key=per_fish_h.get)
    logger.info(f"WEAKEST LINK: fish{worst} at ~{per_fish_h[worst]}s (the look-alike that fails first). "
                f"Beyond ~{H}s, don't trust identity from appearance alone.")
    logger.info(f"saved report -> {out}")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        dv = yaml.safe_load(f)["finetune_reid"]["video"]
    p = argparse.ArgumentParser(description="Re-ID quality: individual recognition, hold-time, confusion")
    p.add_argument("--video_name", default=dv)
    p.add_argument("--stretch", default="04")
    main(p.parse_args().video_name, p.parse_args().stretch)
