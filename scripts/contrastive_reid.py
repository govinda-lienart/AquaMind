"""
contrastive_reid.py — Stage 6.5 SELF-SUPERVISED CONTRASTIVE re-ID (idtracker.ai 2025 method).

THE HYPOTHESIS: raw frozen DINOv2 features do NOT cluster the 5 fish across the whole video
(stitch_ids piece 3: silhouette 0.213, total mush). Generic features encode nuisance (pose/lighting/
tank-region) more than identity. This script LEARNS identity-discriminative features — the lever the
frozen baseline lacks — WITHOUT needing clean global labels, by exploiting the fragment structure:

  - POSITIVE pair = two crops from the SAME fragment  -> guaranteed same fish (a clean run)
  - NEGATIVE pair = crops from COEXISTING fragments   -> guaranteed different fish (can't be two
                    places at once). We build each batch from fragments active at ONE random frame t,
                    so every cross-fragment pair in the batch is a valid negative, and NO two fragments
                    of the same fish are ever pushed apart (same fish can't be at t twice).

Why this can beat the 6.4 fine-tune (which OVERFIT): it trains on ALL 157 fragments spanning the WHOLE
video (max diversity — the exact thing 6.4's single stretch lacked) and needs no swap-prone labels.

Design: trains a small PROJECTION HEAD (MLP) on CACHED frozen DINOv2 features (backbone stays frozen ->
fast, low overfit risk). If a learned projection can surface identity from frozen feats, we win cheaply;
if not, the next escalation is unfreezing the backbone. Loss = supervised contrastive (SupCon) with the
fragment as the label, computed per coexisting group.

Evaluate the SAME way as the frozen baseline (stitch_ids piece 3): average each fragment's learned
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

from scripts.reid_features import load_backbone, transform
from scripts.stitch_ids import flag_clean, cluster_fragments, plot_fragment_clusters, MIN_SEPARATION_PX
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)


# ── frozen features for every clean crop, tagged with its fragment (cached — backbone pass runs once) ──
def load_crop_features(run_dir, frags, backbone_name, device, batch=128):
    """Return (feats (N,384) cpu tensor, crop_frag (N,) int) — one row per clean crop, tagged with the
    fragments.csv row index it belongs to. Cached to stitch/contrastive_cache.npz so tuning re-runs are fast."""
    cache = os.path.join(run_dir, "stitch", "contrastive_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        logger.info(f"loaded cached crop features {z['feats'].shape} from {cache}")
        return torch.from_numpy(z["feats"]).float(), z["crop_frag"]

    tracks = flag_clean(pd.read_parquet(os.path.join(run_dir, "tracks.parquet")), MIN_SEPARATION_PX)
    clean = tracks[tracks["clean"]]
    crops_dir = os.path.join(run_dir, "crops")
    paths, crop_frag = [], []
    for i, fr in frags.iterrows():                                # map each clean crop to its fragment row
        fid = int(fr["fish_id"])
        sub = clean[(clean["fish_id"] == fid) &
                    (clean["frame_number"] >= fr["frame_start"]) & (clean["frame_number"] <= fr["frame_end"])]
        for fn in sub["frame_number"]:
            p = os.path.join(crops_dir, f"fish_{fid}", f"frame_{int(fn)}_fish_{fid}.jpg")
            if os.path.exists(p):
                paths.append(p); crop_frag.append(i)
    logger.info(f"{len(paths)} clean crops across {len(frags)} fragments — running backbone once")

    backbone = load_backbone(backbone_name, device)
    feats = []
    with torch.no_grad():
        for j in range(0, len(paths), batch):
            imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[j:j + batch]]).to(device)
            feats.append(backbone(imgs).cpu())
    feats = torch.cat(feats).float()
    crop_frag = np.array(crop_frag)
    np.savez(cache, feats=feats.numpy().astype("float32"), crop_frag=crop_frag)
    logger.info(f"cached crop features -> {cache}")
    return feats, crop_frag


class Projection(nn.Module):
    """384-d frozen feature -> normalized low-d identity embedding (MLP projection head)."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)                    # unit vectors -> cosine world


def supcon_loss(z, labels, temp):
    """Supervised contrastive loss (Khosla et al.). Pull same-label (same fragment) together, push all
    others (coexisting = different fish) apart. Assumes the batch is ONE coexisting group, so different
    labels are always different fish."""
    sim = (z @ z.T) / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()      # numerical stability
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask  # same fragment, not self
    pos_counts = pos_mask.sum(1)
    valid = pos_counts > 0
    loss = -(pos_mask * log_prob).sum(1)[valid] / pos_counts[valid]
    return loss.mean() if valid.any() else None


def train(feats, crop_frag, frag_range, cfg, device):
    """Train the projection head on coexisting-group batches. Returns the trained head."""
    frag_to_idx = {f: np.where(crop_frag == f)[0] for f in frag_range if (crop_frag == f).sum() >= 2}
    frag_ids = list(frag_to_idx)
    head = Projection(feats.shape[1], cfg["hidden_dim"], cfg["out_dim"]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(0)
    feats = feats.to(device)

    banner_sub(f"training projection head ({cfg['steps']} steps, out_dim={cfg['out_dim']})")
    running, n = 0.0, 0
    for step in range(1, cfg["steps"] + 1):
        A = frag_ids[rng.integers(len(frag_ids))]                 # anchor fragment
        s, e = frag_range[A]
        t = int(rng.integers(s, e + 1))                           # a frame inside it
        active = [f for f in frag_ids if frag_range[f][0] <= t <= frag_range[f][1]]   # fragments alive at t = coexisting = different fish
        if len(active) < 2:
            continue
        idxs, labels = [], []
        for f in active:
            pool = frag_to_idx[f]
            pick = rng.choice(pool, size=min(cfg["crops_per_frag"], len(pool)), replace=False)
            idxs.extend(pick.tolist()); labels.extend([f] * len(pick))
        z = head(feats[idxs])
        loss = supcon_loss(z, torch.tensor(labels, device=device), cfg["temperature"])
        if loss is None:
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item(); n += 1
        if step % 500 == 0:
            logger.info(f"step {step}/{cfg['steps']}  avg loss {running / max(n, 1):.4f}")
            running, n = 0.0, 0
    return head


def embed_fragments(head, feats, crop_frag, n_frags, device):
    """Project all crops, average per fragment -> one learned unit-fingerprint per fragment (n_frags, out_dim)."""
    head.eval()
    with torch.no_grad():
        z = head(feats.to(device)).cpu()
    fps = []
    for f in range(n_frags):
        m = crop_frag == f
        fp = F.normalize(z[m].mean(dim=0), dim=0) if m.any() else torch.zeros(z.shape[1])
        fps.append(fp.numpy())
    return np.stack(fps).astype("float32")


def main(video_name):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    cfg = full["contrastive_reid"]
    run_dir = full["train_reid"]["videos"][video_name]["crops_run"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    banner(f"CONTRASTIVE re-ID (self-supervised, fragment pairs) — {video_name} on {device}")
    frags = pd.read_csv(os.path.join(run_dir, "stitch", "fragments.csv"))
    n_fish = int(frags["fish_id"].nunique())
    frag_range = {i: (int(r["frame_start"]), int(r["frame_end"])) for i, r in frags.iterrows()}

    feats, crop_frag = load_crop_features(run_dir, frags, cfg["backbone"], device)
    head = train(feats, crop_frag, frag_range, cfg, device)

    # ---------- evaluate: learned embeddings -> cluster (SAME protocol as the frozen baseline) ----------
    banner_sub("evaluate — cluster the LEARNED fragment embeddings into n_fish")
    fps = embed_fragments(head, feats, crop_frag, len(frags), device)
    frags, sil = cluster_fragments(frags, fps, n_fish)
    logger.info(f"CONTRASTIVE silhouette: {sil:.3f}   (frozen baseline was 0.213 — did it lift?)")
    logger.info("cluster vs tracker label (fragment counts):\n" +
                pd.crosstab(frags["fish_id"], frags["cluster"]).to_string())

    out_dir = os.path.join(run_dir, "stitch")
    plot_fragment_clusters(frags, fps, os.path.join(out_dir, "fragment_clusters_contrastive.png"))
    frags.to_csv(os.path.join(out_dir, "fragments_contrastive.csv"), index=False)
    torch.save({"head_state": head.state_dict(), "in_dim": feats.shape[1],
                "hidden_dim": cfg["hidden_dim"], "out_dim": cfg["out_dim"], "backbone": cfg["backbone"]},
               os.path.join(out_dir, "contrastive_head.pt"))
    logger.info(f"saved plot + fragments + head -> {out_dir}")
    banner("VERDICT")
    logger.info(f"silhouette {sil:.3f} vs frozen 0.213 · read the contingency: near block-diagonal = fish now separate.")
    logger.info("lifted a lot -> contrastive cracked it (build piece 4 render). flat/mush -> escalate: unfreeze backbone, or it's a single-session DATA limit.")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Self-supervised contrastive re-ID on fragment pairs")
    parser.add_argument("--video_name", default=default_video)
    args = parser.parse_args()
    main(args.video_name)
