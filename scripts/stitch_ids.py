"""
stitch_ids.py — Stage 6 IDENTITY STITCHER (idtracker.ai-style "recognise, don't trace", offline pass 2).

Runs AFTER the tracker. Reads a tracker run's tracks.parquet + crops, and will (piece by piece):
  1. TRACKLET BUILDER      <-- THIS FILE (first piece)
  2. fingerprint each tracklet   (reuse reid_features.build_features + average)
  3. cluster tracklets -> consistent global IDs   (KMeans — proven in stitch_proof.py)
  4. relabel tracks.parquet + render a corrected BEFORE/AFTER video

PIECE 1 — TRACKLET BUILDER
A "tracklet" = a run of frames where ONE fish is safely alone (far from every other fish AND a real
detection, not a ghost). Between two crossings the tracker's ID is trustworthy, so every crop in a
tracklet is guaranteed the same individual — these are the automatic version of the hand-picked
"stretches" used in stitch_proof.py.

Reuses your curate_crops.filter_separated math: per frame, each fish's nearest-neighbour distance.
  - curate_crops: nearest < MIN_SEPARATION_PX  -> DROP the crop (contaminated)
  - here:         nearest < MIN_SEPARATION_PX  -> this frame is a CUT point (a crossing)
                  nearest >= MIN_SEPARATION_PX AND real detection -> frame is INSIDE a tracklet

Usage:  python -m scripts.stitch_ids --video_name IMG_1839


CONFIG

i need to update this part

tracker_crop_parquet:
  default: IMG_1839
  videos:
    IMG_1839:
      run_dir:  output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202
      backbone: dinov2_vits14

e.g python -m scripts.stitch_ids --video_name IMG_2349

"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")                                     # headless — save PNG, never open a window
import matplotlib.pyplot as plt

from scripts.reid_features import load_backbone, transform   # shared DINOv2 embedder + preprocessing belt
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)

MIN_SEPARATION_PX   = 150   # a fish nearer than this to another = a crossing (same value as curate_crops)
MAX_GAP_FRAMES      = 5     # allow tiny gaps (brief 1-frame misses) INSIDE one tracklet; a bigger gap = cut
MIN_TRACKLET_FRAMES = 30    # drop tracklets shorter than this (idtracker.ai floor ~30 imgs/individual)


def flag_clean(tracks, min_dist):
    """Add a boolean 'clean' column: this (frame, fish) is a REAL detection AND far from all other fish.
    Per-frame nearest-neighbour distance — the exact computation from curate_crops.filter_separated."""
    tracks = tracks.copy()
    separated = np.zeros(len(tracks), dtype=bool)
    for _, g in tracks.groupby("frame_number"):
        pos = g[["x", "y"]].to_numpy()
        if len(pos) == 1:
            sep = np.array([True])                       # a lone fish can't be crossing anyone
        else:
            diff = pos[:, None, :] - pos[None, :, :]      # (n, n, 2) pairwise offsets
            dist = np.hypot(diff[..., 0], diff[..., 1])   # (n, n) pairwise distances (same np.hypot as speed)
            np.fill_diagonal(dist, np.inf)                # ignore self
            sep = dist.min(axis=1) >= min_dist            # nearest other fish is far enough
        separated[tracks.index.get_indexer(g.index)] = sep
    tracks["separated"] = separated
    tracks["clean"] = tracks["separated"] & (tracks["occluded"] == 0)   # far AND a real box (not a ghost)
    return tracks


def build_tracklets(tracks, max_gap, min_frames):
    """Cut each fish's clean frames into tracklets (runs broken by crossings or big gaps).
    Returns a DataFrame: fish_id | frame_start | frame_end | n_frames."""
    rows = []
    clean = tracks[tracks["clean"]]
    for fid, g in clean.groupby("fish_id"):
        frames = np.sort(g["frame_number"].to_numpy())
        if len(frames) == 0:
            continue
        cuts = np.where(np.diff(frames) > max_gap)[0]     # index after which the run breaks
        starts = np.concatenate([[0], cuts + 1])          # segment start indices
        ends = np.concatenate([cuts, [len(frames) - 1]])  # segment end indices
        for s, e in zip(starts, ends):
            n = e - s + 1
            if n >= min_frames:                           # long enough to trust / enough crops
                rows.append({"fish_id": int(fid),
                             "frame_start": int(frames[s]),
                             "frame_end": int(frames[e]),
                             "n_frames": int(n)})
    tracklets = pd.DataFrame(rows, columns=["fish_id", "frame_start", "frame_end", "n_frames"])
    return tracklets.sort_values(["frame_start", "fish_id"]).reset_index(drop=True)


def fingerprint_tracklets(tracklets, tracks, run_dir, backbone_name, device, batch=128):
    """One averaged unit-fingerprint per tracklet: load its CLEAN-frame crops -> backbone -> mean -> L2-norm.
    Returns (tracklets_kept, fps) where fps is (n_tracklets, D) float32, row-aligned with tracklets_kept.
    A tracklet with no crops on disk is dropped (keeps rows and fingerprints aligned)."""
    backbone = load_backbone(backbone_name, device)
    crops_dir = os.path.join(run_dir, "crops")
    clean = tracks[tracks["clean"]]
    clean_by_fish = {int(fid): set(g["frame_number"].tolist()) for fid, g in clean.groupby("fish_id")}

    fps, keep_idx = [], []
    for i, trk in tracklets.iterrows():
        fid, fs, fe = int(trk["fish_id"]), trk["frame_start"], trk["frame_end"]
        frames = sorted(n for n in clean_by_fish[fid] if fs <= n <= fe)          # this tracklet's clean frames
        paths = [os.path.join(crops_dir, f"fish_{fid}", f"frame_{n}_fish_{fid}.jpg") for n in frames]
        paths = [p for p in paths if os.path.exists(p)]                          # a few clean frames may lack a saved crop
        if not paths:
            logger.warning(f"tracklet fish{fid} {fs}-{fe}: no crops on disk — dropped")
            continue
        chunks = []
        with torch.no_grad():                                                    # frozen backbone — no gradients
            for j in range(0, len(paths), batch):
                imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[j:j + batch]]).to(device)
                chunks.append(backbone(imgs).cpu())
        crop_feats = F.normalize(torch.cat(chunks), dim=1)                       # unit fingerprint per crop
        fp = F.normalize(crop_feats.mean(dim=0), dim=0)                          # average -> the tracklet fingerprint
        fps.append(fp.numpy())
        keep_idx.append(i)
    tracklets_kept = tracklets.loc[keep_idx].reset_index(drop=True)
    return tracklets_kept, np.stack(fps).astype("float32")


def cluster_tracklets(tracklets, fps, n_fish):
    """Group the tracklet fingerprints into n_fish clusters = the true physical fish.
    This OVERRIDES the tracker's per-tracklet labels (which carry silent swaps).
    Returns (tracklets + 'cluster' column, silhouette score)."""
    labels = KMeans(n_clusters=n_fish, n_init=10, random_state=0).fit_predict(fps)
    tracklets = tracklets.copy()
    tracklets["cluster"] = labels
    sil = silhouette_score(fps, labels) if len(set(labels)) > 1 else float("nan")
    return tracklets, sil


def plot_tracklet_clusters(tracklets, fps, out_path):
    """Two t-SNE panels of the tracklet fingerprints — LEFT coloured by the stitcher's cluster,
    RIGHT by the tracker's label. Agreement = both right; divergence = a tracker swap OR a cluster error."""
    emb = TSNE(n_components=2, init="pca", perplexity=min(30, len(fps) - 1), random_state=0).fit_transform(fps)
    sizes = 20 + 180 * (tracklets["n_frames"] / tracklets["n_frames"].max())      # bigger dot = longer tracklet
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, title in [(axes[0], "cluster", "stitcher k-means cluster (the answer)"),
                           (axes[1], "fish_id", "tracker label (noisy reference)")]:
        for k, v in enumerate(sorted(tracklets[key].unique())):
            m = (tracklets[key] == v).to_numpy()
            ax.scatter(emb[m, 0], emb[m, 1], s=sizes[m], alpha=0.7, color=plt.cm.tab10(k), label=f"{key} {v}")
        ax.set_title(title); ax.legend(fontsize=8, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{len(fps)} tracklet fingerprints — LEFT: stitcher clusters | RIGHT: tracker labels "
                 "(dot size = tracklet length)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"saved cluster plot -> {out_path}")


def main(video_name):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["tracker_crop_parquet"]   # tracker run dir (has tracks.parquet + crops)
    run_dir = cfg["videos"][video_name]["run_dir"]

    banner(f"STITCH — piece 1: TRACKLET BUILDER  ({video_name})")
    tracks = pd.read_parquet(os.path.join(run_dir, "tracks.parquet"))
    n_fish = int(tracks.fish_id.nunique())
    logger.info(f"tracks: {len(tracks)} rows | fish {sorted(tracks.fish_id.unique().tolist())} "
                f"| frames {int(tracks.frame_number.min())}–{int(tracks.frame_number.max())}")

    banner_sub("flag clean frames (far from all others AND a real detection)")
    tracks = flag_clean(tracks, MIN_SEPARATION_PX)
    n_clean, n_total = int(tracks["clean"].sum()), len(tracks)
    logger.info(f"clean (tracklet-worthy) rows: {n_clean}/{n_total} ({100*n_clean/n_total:.0f}%) "
                f"— the rest are crossings or ghost frames")

    banner_sub(f"cut into tracklets (max_gap={MAX_GAP_FRAMES}, min_frames={MIN_TRACKLET_FRAMES})")
    tracklets = build_tracklets(tracks, MAX_GAP_FRAMES, MIN_TRACKLET_FRAMES)
    logger.info(f"built {len(tracklets)} tracklets across {tracklets.fish_id.nunique()} fish")
    logger.info("tracklets per fish:\n" + tracklets.groupby("fish_id").size().to_string())
    logger.info("total clean frames captured per fish:\n" + tracklets.groupby("fish_id")["n_frames"].sum().to_string())
    banner_sub("all tracklets (fish_id | frame_start | frame_end | n_frames)")
    logger.info("\n" + tracklets.to_string(index=False))

    out_dir = os.path.join(run_dir, "stitch")
    os.makedirs(out_dir, exist_ok=True)
    tracklets.to_csv(os.path.join(out_dir, "tracklets.csv"), index=False)
    logger.info(f"saved tracklets -> {os.path.join(out_dir, 'tracklets.csv')}")

    # ---------- piece 2: one fingerprint per tracklet ----------
    backbone_name = cfg["videos"][video_name]["backbone"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    banner_sub(f"fingerprint each tracklet (backbone {backbone_name} on {device})")
    tracklets, fps = fingerprint_tracklets(tracklets, tracks, run_dir, backbone_name, device)
    logger.info(f"fingerprinted {len(tracklets)} tracklets -> fps {fps.shape}  (each row = one tracklet's average look)")
    np.save(os.path.join(out_dir, "tracklet_fps.npy"), fps)                      # aligned row-for-row with tracklets.csv
    logger.info(f"saved fingerprints -> {os.path.join(out_dir, 'tracklet_fps.npy')}")

    # ---------- piece 3: cluster tracklets into global IDs (the moment of truth) ----------
    banner_sub(f"cluster {len(tracklets)} tracklets into {n_fish} fish (k-means) — this OVERRIDES tracker labels")
    tracklets, sil = cluster_tracklets(tracklets, fps, n_fish)
    logger.info(f"silhouette score: {sil:.3f}  (>0.5 strong · 0.25–0.5 workable · <0.25 weak/overlapping)")

    banner_sub("cluster vs tracker label  (rows = tracker fish_id, cols = cluster; tracker labels are a NOISY reference, not truth)")
    logger.info("tracklet counts:\n" + pd.crosstab(tracklets["fish_id"], tracklets["cluster"]).to_string())
    logger.info("frame-weighted (a long tracklet counts more):\n" +
                pd.crosstab(tracklets["fish_id"], tracklets["cluster"], values=tracklets["n_frames"], aggfunc="sum").fillna(0).astype(int).to_string())

    tracklets.to_csv(os.path.join(out_dir, "tracklets.csv"), index=False)            # now with 'cluster' column
    plot_tracklet_clusters(tracklets, fps, os.path.join(out_dir, "tracklet_clusters.png"))
    logger.info(f"saved tracklets (with cluster) -> {os.path.join(out_dir, 'tracklets.csv')}")
    logger.info("READ IT: a near block-diagonal table = clusters agree with the tracker (both right). "
                "Off-diagonal = a tracker swap the stitcher FIXED, or a cluster error to inspect. "
                "fish3/5 sharing a cluster = the look-alike wall on the real system.")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["tracker_crop_parquet"]["default"]
    parser = argparse.ArgumentParser(description="Identity stitcher — piece 1: build tracklets from tracks.parquet")
    parser.add_argument("--video_name", default=default_video, help="key under tracker_crop_parquet.videos in config.yaml")
    args = parser.parse_args()
    main(args.video_name)
