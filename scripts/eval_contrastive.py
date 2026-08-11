"""
eval_contrastive.py — DECISIVE, label-trustworthy test of the contrastive embedding.

The whole-video silhouette (0.512) is ambiguous: it says "5 tight clusters exist" but the tracker labels
are too swap-ridden to confirm those clusters are the 5 FISH. So test on a stretch where identity IS
known: stretch04 is one clean continuous stretch → its per-fish labels are trustworthy (no cross-tracklet
swaps within it). We ask, on TRUE labels: does the contrastive embedding separate the 5 fish BETTER than
raw frozen DINOv2 — especially the fish3/5 look-alikes raw couldn't split?

Metrics (raw frozen  vs  contrastive), both on the SAME trusted labels:
  - kNN identity accuracy: for each crop, is its nearest neighbour the same fish? (direct separation score)
  - silhouette on TRUE labels: how tight/separated the 5 real fish are
  - per-fish kNN accuracy: exposes fish3/5 specifically
Plus a before/after t-SNE coloured by TRUE fish.

CAVEAT: these crops were in the contrastive TRAINING data — but their LABELS were not (training is
self-supervised on tracklet pairs, never on fish labels). So this measures whether the learned features
separate identity, on labels the model never saw. Not a fully held-out test (no unseen stretch exists),
but a fair read of representation quality.

Usage:  python -m scripts.eval_contrastive --video_name IMG_1839 --stretch 04
"""
import os
import argparse
import logging

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reid_features import build_features
from scripts.contrastive_reid import Projection
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)


def knn_acc(emb, labels):
    """Leave-one-out nearest-neighbour identity accuracy: is each crop's closest OTHER crop the same fish?"""
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1.0)
    nn = sim.argmax(dim=1)
    return (labels[nn] == labels).float().mean().item(), (labels[nn] == labels)


def report(name, emb, labels, slot_to_fish):
    acc, hit = knn_acc(emb, labels)
    sil = silhouette_score(F.normalize(emb, dim=1).numpy(), labels.numpy())
    logger.info(f"[{name}]  kNN identity acc: {acc:.3f}   silhouette(true labels): {sil:.3f}")
    for slot in sorted(labels.unique().tolist()):
        m = labels == slot
        logger.info(f"    fish{slot_to_fish[slot]}: kNN acc {hit[m].float().mean().item():.3f}  (n={int(m.sum())})")
    return acc, sil


def main(video_name, stretch):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    run_dir = full["train_reid"]["videos"][video_name]["crops_run"]
    backbone = full["contrastive_reid"]["backbone"]

    banner(f"CONTRASTIVE vs RAW — trusted-label test on stretch {stretch} ({video_name})")
    feats, labels, frames, label_map = build_features(run_dir, [stretch], backbone)
    slot_to_fish = {slot: fish for fish, slot in label_map.items()}
    logger.info(f"{len(feats)} crops | {len(label_map)} fish (labels are TRUSTED within this stretch)")

    # ---------- contrastive projection ----------
    ck = torch.load(os.path.join(run_dir, "stitch", "contrastive_head.pt"), map_location="cpu")
    head = Projection(ck["in_dim"], ck["hidden_dim"], ck["out_dim"])
    head.load_state_dict(ck["head_state"]); head.eval()
    with torch.no_grad():
        contrastive = head(feats)

    out_dir = os.path.join(run_dir, "stitch")
    log_path = os.path.join(out_dir, f"eval_contrastive_stretch{stretch}.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(funcName)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)               # mirrors console output into the run's stitch/ folder

    banner_sub("RAW frozen DINOv2  (the baseline)")
    raw_acc, raw_sil = report("raw", feats, labels, slot_to_fish)
    banner_sub("CONTRASTIVE embedding  (the learned features)")
    con_acc, con_sil = report("contrastive", contrastive, labels, slot_to_fish)

    # ---------- before/after t-SNE coloured by TRUE fish ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, emb, title in [(axes[0], feats, f"RAW frozen (kNN {raw_acc:.2f}, silhouette {raw_sil:.3f})"),
                           (axes[1], contrastive, f"CONTRASTIVE (kNN {con_acc:.2f}, silhouette {con_sil:.3f})")]:
        xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(F.normalize(emb, dim=1).numpy())
        for k, slot in enumerate(sorted(labels.unique().tolist())):
            m = (labels == slot).numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.35, color=plt.cm.tab10(k), label=f"fish{slot_to_fish[slot]}")
        ax.set_title(title); ax.legend(markerscale=2, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"stretch {stretch}, coloured by TRUE fish — LEFT raw | RIGHT contrastive (clean islands incl. fish3/5 = it worked)")
    fig.tight_layout()
    out = os.path.join(out_dir, f"eval_contrastive_stretch{stretch}.png")
    fig.savefig(out, dpi=130); plt.close(fig)

    banner("VERDICT")
    logger.info(f"kNN identity acc:  raw {raw_acc:.3f}  ->  contrastive {con_acc:.3f}   (higher = fish better separated)")
    logger.info(f"silhouette(true):  raw {raw_sil:.3f}  ->  contrastive {con_sil:.3f}")
    logger.info(f"saved before/after t-SNE -> {out}")
    logger.info(f"saved run log -> {log_path}")
    logger.info("big rise + fish3/5 acc up -> contrastive learned IDENTITY (whole-video 0.512 was real). "
                "flat -> the 0.512 was nuisance structure -> escalate (unfreeze backbone) or single-session data limit.")
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Trusted-label test: does the contrastive embedding separate fish better than raw?")
    parser.add_argument("--video_name", default=default_video)
    parser.add_argument("--stretch", default="04")
    args = parser.parse_args()
    main(args.video_name, args.stretch)
