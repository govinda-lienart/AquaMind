"""
contrastive_reid.py — Stage 6.5 SELF-SUPERVISED CONTRASTIVE re-ID (idtracker.ai 2025 method).

THE HYPOTHESIS: raw frozen DINOv2 features do NOT cluster the 5 fish across the whole video
(stitch_ids piece 3: silhouette 0.213, total mush). Generic features encode nuisance (pose/lighting/
tank-region) more than identity. This script LEARNS identity-discriminative features — the lever the
frozen baseline lacks — WITHOUT needing clean global labels, by exploiting the tracklet structure:

  - POSITIVE pair = two crops from the SAME tracklet  -> guaranteed same fish (a clean run)
  - NEGATIVE pair = crops from COEXISTING tracklets    -> guaranteed different fish (can't be two
                    places at once). We build each batch from tracklets active at ONE random frame t,
                    so every cross-tracklet pair in the batch is a valid negative, and NO two tracklets
                    of the same fish are ever pushed apart (same fish can't be at t twice).

Why this can beat the 6.4 fine-tune (which OVERFIT): it trains on ALL 157 tracklets spanning the WHOLE
video (max diversity — the exact thing 6.4's single stretch lacked) and needs no swap-prone labels.

Design: trains a small PROJECTION HEAD (MLP) on CACHED frozen DINOv2 features (backbone stays frozen ->
fast, low overfit risk). If a learned projection can surface identity from frozen feats, we win cheaply;
if not, the next escalation is unfreezing the backbone. Loss = supervised contrastive (SupCon) with the
tracklet as the label, computed per coexisting group.

Evaluate the SAME way as the frozen baseline (stitch_ids piece 3): average each tracklet's learned
embeddings -> k-means into n_fish -> silhouette + contingency + t-SNE. Compare against raw's 0.213.

Usage:  python -m scripts.contrastive_reid --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reid_features import load_backbone, transform, build_features
from scripts.stitch_ids import flag_clean, cluster_tracklets, plot_tracklet_clusters, MIN_SEPARATION_PX
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)
CHECK_EVERY = 500   # how often (in steps) to log train loss + validation metrics


# ── frozen features for every clean crop, tagged with its tracklet (cached — backbone pass runs once) ──
def load_crop_features(run_dir, tracklets, backbone_name, device, batch=128):
    """Return (feats (N,384) cpu tensor, crop_tracklet (N,) int) — one row per clean crop, tagged with the
    tracklets.csv row index it belongs to. Cached to stitch/contrastive_cache.npz so tuning re-runs are fast."""
    cache = os.path.join(run_dir, "stitch", "contrastive_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        logger.info(f"loaded cached crop features {z['feats'].shape} from {cache}")
        return torch.from_numpy(z["feats"]).float(), z["crop_tracklet"]

    tracks = flag_clean(pd.read_parquet(os.path.join(run_dir, "tracks.parquet")), MIN_SEPARATION_PX)
    clean = tracks[tracks["clean"]]
    crops_dir = os.path.join(run_dir, "crops")
    paths, crop_tracklet = [], []
    for i, tr in tracklets.iterrows():                            # map each clean crop to its tracklet row
        fid = int(tr["fish_id"])
        sub = clean[(clean["fish_id"] == fid) &
                    (clean["frame_number"] >= tr["frame_start"]) & (clean["frame_number"] <= tr["frame_end"])]
        for fn in sub["frame_number"]:
            p = os.path.join(crops_dir, f"fish_{fid}", f"frame_{int(fn)}_fish_{fid}.jpg")
            if os.path.exists(p):
                paths.append(p); crop_tracklet.append(i)
    logger.info(f"{len(paths)} clean crops across {len(tracklets)} tracklets — running backbone once")

    backbone = load_backbone(backbone_name, device)
    feats = []
    with torch.no_grad():
        for j in range(0, len(paths), batch):
            imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[j:j + batch]]).to(device)
            feats.append(backbone(imgs).cpu())
    feats = torch.cat(feats).float()
    crop_tracklet = np.array(crop_tracklet)
    np.savez(cache, feats=feats.numpy().astype("float32"), crop_tracklet=crop_tracklet)
    logger.info(f"cached crop features -> {cache}")
    return feats, crop_tracklet


class Projection(nn.Module):
    """384-d frozen feature -> normalized low-d identity embedding (MLP projection head)."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)                    # unit vectors -> cosine world


def supcon_loss(z, labels, temp):
    """Supervised contrastive loss (Khosla et al.). Pull same-label (same tracklet) together, push all
    others (coexisting = different fish) apart. Assumes the batch is ONE coexisting group, so different
    labels are always different fish."""
    sim = (z @ z.T) / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()      # numerical stability
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask  # same tracklet, not self
    pos_counts = pos_mask.sum(1)
    valid = pos_counts > 0
    loss = -(pos_mask * log_prob).sum(1)[valid] / pos_counts[valid]
    return loss.mean() if valid.any() else None


def validate(head, val_feats, val_labels, device):
    """Project the TRUSTED-stretch validation crops through the head AS IT IS RIGHT NOW mid-training.
    Returns (kNN identity accuracy, silhouette on true labels) — the metric-learning equivalent of a
    val loss curve. Does not touch training state (head returns to train mode before we leave)."""
    head.eval()
    with torch.no_grad():
        z = head(val_feats.to(device)).cpu()
    head.train()
    emb = F.normalize(z, dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)
    acc = (val_labels[nn_idx] == val_labels).float().mean().item()
    sil = silhouette_score(emb.numpy(), val_labels.numpy()) if len(set(val_labels.tolist())) > 1 else float("nan")
    return acc, sil


def train(feats, crop_tracklet, tracklet_range, cfg, device, val_feats=None, val_labels=None):
    """Train the projection head on coexisting-group batches.
    Returns (trained head, history) where history has 'steps', 'train_loss', and — if a trusted
    validation set was passed — 'val_knn'/'val_sil' too, logged every CHECK_EVERY steps."""
    tracklet_to_idx = {f: np.where(crop_tracklet == f)[0] for f in tracklet_range if (crop_tracklet == f).sum() >= 2}
    tracklet_ids = list(tracklet_to_idx)
    head = Projection(feats.shape[1], cfg["hidden_dim"], cfg["out_dim"]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(0)
    feats = feats.to(device)
    history = {"steps": [], "train_loss": [], "val_knn": [], "val_sil": []}

    banner_sub(f"training projection head ({cfg['steps']} steps, out_dim={cfg['out_dim']})")
    running, n = 0.0, 0
    for step in range(1, cfg["steps"] + 1):
        A = tracklet_ids[rng.integers(len(tracklet_ids))]         # anchor tracklet
        s, e = tracklet_range[A]
        t = int(rng.integers(s, e + 1))                           # a frame inside it
        active = [f for f in tracklet_ids if tracklet_range[f][0] <= t <= tracklet_range[f][1]]   # tracklets alive at t = coexisting = different fish
        if len(active) < 2:
            continue
        idxs, labels = [], []
        for f in active:
            pool = tracklet_to_idx[f]
            pick = rng.choice(pool, size=min(cfg["crops_per_tracklet"], len(pool)), replace=False)
            idxs.extend(pick.tolist()); labels.extend([f] * len(pick))
        z = head(feats[idxs])
        loss = supcon_loss(z, torch.tensor(labels, device=device), cfg["temperature"])
        if loss is None:
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item(); n += 1
        if step % CHECK_EVERY == 0:
            avg_loss = running / max(n, 1)
            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            msg = f"step {step}/{cfg['steps']}  avg loss {avg_loss:.4f}"
            if val_feats is not None:
                acc, sil = validate(head, val_feats, val_labels, device)
                history["val_knn"].append(acc)
                history["val_sil"].append(sil)
                msg += f"  |  trusted-stretch kNN {acc:.3f}  silhouette {sil:.3f}"
            logger.info(msg)
            running, n = 0.0, 0
    return head, history


def plot_training_curve(history, out_path):
    """Loss + validation curves vs step — the metric-learning equivalent of YOLO's results.png."""
    has_val = len(history["val_knn"]) > 0
    fig, axes = plt.subplots(1, 3 if has_val else 1, figsize=(15 if has_val else 5.5, 4.5))
    axes = np.atleast_1d(axes)
    axes[0].plot(history["steps"], history["train_loss"], "o-", color="tab:blue")
    axes[0].set_title("train/supcon_loss"); axes[0].set_xlabel("step"); axes[0].grid(alpha=.3)
    if has_val:
        axes[1].plot(history["steps"], history["val_knn"], "o-", color="tab:green")
        axes[1].set_title("val/kNN accuracy (trusted stretch)"); axes[1].set_xlabel("step")
        axes[1].set_ylim(0, 1.02); axes[1].grid(alpha=.3)
        axes[2].plot(history["steps"], history["val_sil"], "o-", color="tab:orange")
        axes[2].set_title("val/silhouette (trusted stretch)"); axes[2].set_xlabel("step"); axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
    logger.info(f"saved training curve -> {out_path}")


def embed_tracklets(head, feats, crop_tracklet, n_tracklets, device):
    """Project all crops, average per tracklet -> one learned unit-fingerprint per tracklet (n_tracklets, out_dim)."""
    head.eval()
    with torch.no_grad():
        z = head(feats.to(device)).cpu()
    fps = []
    for f in range(n_tracklets):
        m = crop_tracklet == f
        fp = F.normalize(z[m].mean(dim=0), dim=0) if m.any() else torch.zeros(z.shape[1])
        fps.append(fp.numpy())
    return np.stack(fps).astype("float32")


def main(video_name, stretch="04"):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    cfg = full["contrastive_reid"]
    run_info = full["tracker_crop_parquet"]["videos"][video_name]
    run_dir = run_info["run_dir"]
    backbone_name = run_info["backbone"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    out_dir = os.path.join(run_dir, "stitch")
    log_path = os.path.join(out_dir, "contrastive_training.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(funcName)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)               # persists this run's log next to the head/plots

    banner(f"CONTRASTIVE re-ID (self-supervised, tracklet pairs) — {video_name} on {device}")
    tracklets = pd.read_csv(os.path.join(run_dir, "stitch", "tracklets.csv"))
    n_fish = int(tracklets["fish_id"].nunique())
    tracklet_range = {i: (int(r["frame_start"]), int(r["frame_end"])) for i, r in tracklets.iterrows()}

    feats, crop_tracklet = load_crop_features(run_dir, tracklets, backbone_name, device)

    # ---------- trusted-stretch validation set, checked periodically DURING training (not just at the end) ----------
    val_feats, val_labels = None, None
    try:
        crops_run = full["finetune_reid"]["videos"][video_name]["crops_run"]
        val_feats, val_labels, _, _ = build_features(crops_run, [stretch], backbone_name)
        logger.info(f"validation set: {len(val_feats)} crops from trusted stretch {stretch}")
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        logger.warning(f"no trusted-stretch validation set available ({e}) — training curve will show loss only")

    head, history = train(feats, crop_tracklet, tracklet_range, cfg, device, val_feats, val_labels)
    plot_training_curve(history, os.path.join(out_dir, "contrastive_training_curve.png"))

    # ---------- evaluate: learned embeddings -> cluster (SAME protocol as the frozen baseline) ----------
    banner_sub("evaluate — cluster the LEARNED tracklet embeddings into n_fish")
    fps = embed_tracklets(head, feats, crop_tracklet, len(tracklets), device)
    tracklets, sil = cluster_tracklets(tracklets, fps, n_fish)
    logger.info(f"CONTRASTIVE silhouette: {sil:.3f}   (frozen baseline was 0.213 — did it lift?)")
    logger.info("cluster vs tracker label (tracklet counts):\n" +
                pd.crosstab(tracklets["fish_id"], tracklets["cluster"]).to_string())

    plot_tracklet_clusters(tracklets, fps, os.path.join(out_dir, "tracklet_clusters_contrastive.png"))
    torch.save({"head_state": head.state_dict(), "in_dim": feats.shape[1],
                "hidden_dim": cfg["hidden_dim"], "out_dim": cfg["out_dim"], "backbone": backbone_name},
               os.path.join(out_dir, "contrastive_head.pt"))
    logger.info(f"saved plot + head -> {out_dir}")
    banner("VERDICT")
    logger.info(f"silhouette {sil:.3f} vs frozen 0.213 · read the contingency: near block-diagonal = fish now separate.")
    logger.info("lifted a lot -> contrastive cracked it (build piece 4 render). flat/mush -> escalate: unfreeze backbone, or it's a single-session DATA limit.")
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["tracker_crop_parquet"]["default"]
    parser = argparse.ArgumentParser(description="Self-supervised contrastive re-ID on tracklet pairs")
    parser.add_argument("--video_name", default=default_video)
    parser.add_argument("--stretch", default="04", help="trusted stretch used as a periodic validation set during training")
    args = parser.parse_args()
    main(args.video_name, args.stretch)
