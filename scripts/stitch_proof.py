"""
stitch_proof.py — PROOF-OF-CONCEPT for the idtracker.ai-style "recognise, don't trace" stitcher.

THE ONE QUESTION THIS ANSWERS: can appearance fingerprints REJOIN two separate pieces of the SAME
fish? That is the single new ability the stitcher needs — the tracker already tells fish apart
*within* one continuous piece; the hard part is recognising that a piece here and a piece there are
the same individual, so we can stitch a whole video's fragments back into consistent identities.

TRUSTED-BY-CONSTRUCTION TEST (no cross-stretch labels needed — that's the beauty):
  1. take ONE stretch whose IDs are verified (no swaps) -> every fish is ONE continuous track
  2. split each fish's crops into an EARLY half and a LATE half by frame number
     -> 2 "pseudo-fragments" per fish, GUARANTEED same fish (unbroken track, no crossing between them)
  3. average each half's crop-fingerprints into ONE fingerprint per half
  4. TEST A (nearest-neighbour): is each half's closest OTHER half its own fish's twin? (direct rejoin)
  5. TEST B (clustering): k-means into n_fish groups -> do a fish's two halves land in the SAME cluster?

RAW frozen DINOv2 features ON PURPOSE (NOT the trained head): the head was trained on THIS stretch's
identities, so using it here would LEAK the answer. This tests whether GENERIC fingerprints already
rejoin -> the honest baseline. If they don't, that's the signal the fingerprints need sharpening
(contrastive learning) before the real stitcher can trust them.

Usage:  python -m scripts.stitch_proof --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")                                     # headless — save PNG, never open a window
import matplotlib.pyplot as plt

from scripts.reid_features import build_features
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)


def half_fingerprints(feats, labels, frames):
    """Split each fish into an EARLY/LATE half by frame; return one averaged unit-fingerprint per half.

    Returns:
        fps        (2F, D) tensor — one unit-length fingerprint per half
        true_fish  list len 2F    — the slot (fish) each half really belongs to
        which_half list len 2F    — 'A' (early) or 'B' (late)
    """
    fps, true_fish, which_half = [], [], []
    for fish in labels.unique().tolist():
        mask = labels == fish
        f_feats = F.normalize(feats[mask], dim=1)         # unit-length per crop (cosine world)
        f_frames = frames[mask]
        order = torch.argsort(f_frames)                   # early -> late, within this fish
        f_feats = f_feats[order]
        if len(f_feats) < 2:                              # need at least one crop per half
            logger.warning(f"fish slot {fish}: only {len(f_feats)} crop(s) — skipping (can't halve)")
            continue
        mid = len(f_feats) // 2
        for tag, part in (("A", f_feats[:mid]), ("B", f_feats[mid:])):
            fp = F.normalize(part.mean(dim=0), dim=0)      # average the crops -> renormalize = the fragment fingerprint
            fps.append(fp)
            true_fish.append(fish)
            which_half.append(tag)
    return torch.stack(fps), true_fish, which_half


def plot_embedding(feats, labels, slot_to_fish, fps, names, stretch, out_path):
    """2D t-SNE of every crop (coloured by fish) + the 10 half-fingerprints as stars.
    Lets you SEE it: separate clouds = separable fish; overlapping clouds = look-alikes (fish3/5);
    a fish's two stars far apart = that fish's halves didn't rejoin (fish1)."""
    crops = F.normalize(feats, dim=1).cpu().numpy()       # per-crop unit fingerprints
    cents = fps.cpu().numpy()                             # the 10 half-fingerprints
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        np.vstack([crops, cents]))                        # embed crops + centroids TOGETHER (same space)
    e_crop, e_cent = emb[:len(crops)], emb[len(crops):]
    lab = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.tab10
    for k, fish in enumerate(sorted(set(lab.tolist()))):
        m = lab == fish
        ax.scatter(e_crop[m, 0], e_crop[m, 1], s=8, alpha=0.22, color=cmap(k), label=f"fish{slot_to_fish[fish]}")
    for i, name in enumerate(names):                      # the half-fingerprints as labelled stars
        ax.scatter(e_cent[i, 0], e_cent[i, 1], s=240, marker="*", color="white",
                   edgecolor="black", linewidth=1.3, zorder=5)
        ax.annotate(name, (e_cent[i, 0], e_cent[i, 1]), fontsize=8, weight="bold",
                    xytext=(6, 4), textcoords="offset points", zorder=6)
    ax.set_title(f"Re-ID fingerprints — IMG_1839 stretch {stretch} (frozen DINOv2)\n"
                 f"stars = early/late half-fingerprints; overlapping colours = look-alike fish")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"saved embedding plot -> {out_path}")


def main(video_name):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["train_reid"]             # reuse train_reid's crops_run / stretch / backbone
    v = cfg["videos"][video_name]
    crops_run, stretch, backbone = v["crops_run"], v["stretches"][0], v["backbone"]

    banner(f"STITCH PROOF — can fingerprints REJOIN the same fish?  (video {video_name}, stretch {stretch})")

    # ---------- fingerprint every crop (reuses the exact same backbone pass as train_reid) ----------
    feats, labels, frames, label_map = build_features(crops_run, [stretch], backbone)
    slot_to_fish = {slot: fish for fish, slot in label_map.items()}
    logger.info(f"{len(feats)} crops | {len(label_map)} fish | frames {int(frames.min())}–{int(frames.max())}")

    # ---------- split each fish into early/late halves, one fingerprint each ----------
    fps, true_fish, which_half = half_fingerprints(feats, labels, frames)
    n_fish = len(labels.unique())
    names = [f"fish{slot_to_fish[tf]}-{h}" for tf, h in zip(true_fish, which_half)]

    # ---------- TEST A: nearest-neighbour rejoin (the direct test) ----------
    banner_sub("TEST A — nearest-neighbour rejoin  (is each half's closest OTHER half its own twin?)")
    sim = fps @ fps.T                                     # cosine similarity (all unit vectors)
    sim.fill_diagonal_(-1.0)                              # ignore self-match
    nn = sim.argmax(dim=1)                                # index of each half's closest OTHER half
    rejoined = 0
    for i, name in enumerate(names):
        j = nn[i].item()
        hit = true_fish[i] == true_fish[j]
        rejoined += int(hit)
        logger.info(f"  {name:11s} -> nearest {names[j]:11s}  {'✅ same fish' if hit else '❌ WRONG'}  (cos {sim[i, j].item():.3f})")
    logger.info(f"TEST A RESULT: {rejoined}/{len(names)} halves found their own twin as nearest neighbour")

    # ---------- TEST B: k-means clustering into n_fish groups ----------
    banner_sub(f"TEST B — k-means into {n_fish} clusters  (do a fish's two halves co-cluster?)")
    clusters = KMeans(n_clusters=n_fish, n_init=10, random_state=0).fit_predict(fps.cpu().numpy())
    by_cluster = {}
    for name, c in zip(names, clusters):
        by_cluster.setdefault(int(c), []).append(name)
    for c, members in sorted(by_cluster.items()):
        logger.info(f"  cluster {c}: {members}")
    together = 0
    for fish in labels.unique().tolist():
        idx = [i for i, tf in enumerate(true_fish) if tf == fish]
        if len(idx) < 2:
            continue
        same = clusters[idx[0]] == clusters[idx[1]]
        together += int(same)
        logger.info(f"  fish{slot_to_fish[fish]}: halves {'TOGETHER ✅' if same else 'SPLIT ❌'}")
    logger.info(f"TEST B RESULT: {together}/{n_fish} fish had both halves in the same cluster")

    # ---------- visualise ----------
    banner_sub("PLOT — 2D t-SNE of all crops + half-fingerprints")
    out_dir = os.path.join(crops_run, "stitch_proof")     # keep the figure WITH the crops it describes
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_name}_stretch{stretch}_embedding.png")
    plot_embedding(feats, labels, slot_to_fish, fps, names, stretch, out_path)

    # ---------- verdict ----------
    banner("VERDICT")
    logger.info(f"Nearest-neighbour rejoin: {rejoined}/{len(names)} halves | Clustering co-locate: {together}/{n_fish} fish")
    logger.info("HIGH on both  -> generic fingerprints CAN rejoin -> the stitcher approach works on your fish. Build it.")
    logger.info("fish3/fish5 splitting -> the known look-alike wall -> sharpen fingerprints (contrastive) BEFORE stitching.")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["train_reid"]["video"]
    parser = argparse.ArgumentParser(description="Proof: can appearance fingerprints rejoin the same fish across a gap?")
    parser.add_argument("--video_name", default=default_video, help="key under train_reid.videos in config.yaml")
    args = parser.parse_args()
    main(args.video_name)
