"""
stitch_accumulate_finetune.py — ACCUMULATION with BACKBONE FINE-TUNING (idtracker.ai's actual method).

stitch_accumulate.py trained a linear head on FROZEN features and stalled (30/157) — because the drift
lives IN the frozen features and a linear head can't fix its own input. This version RETRAINS THE FEATURES:
each accumulation round fine-tunes the last N DINOv2 blocks (+ head) on the crop IMAGES, so the network
learns drift-invariant fish features. No frozen cache — features are recomputed from pixels every round
(that's the whole point, and why it's heavy → GPU / Kaggle).

Loop (same accumulation logic as the frozen version, but the model LEARNS each round):
  1. SEED from a global fragment (all N fish separated) -> N identities.
  2. Fine-tune (last N blocks + head) on the labelled fragments' crops (augmented: flip/rotate/brightness,
     NO hue — same-morph pigment rule).
  3. Predict every unlabelled fragment; ACCEPT confident (mean softmax >= thresh) + coexistence-safe.
  4. Retrain on seed + accepted; repeat until nothing new is added.
  5. Assign leftovers by the final model (respect coexistence).

Writes fragments_stitched.csv (column 'cluster') -> then re-render / re-timeline as usual.
HEAVY: minutes/round on MPS. Sanity-check on a stretch or move to Kaggle's NVIDIA GPU for the full video.

Usage:  python -m scripts.stitch_accumulate_finetune --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from scripts.reid_features import load_backbone, transform          # eval preprocessing belt + backbone loader
from scripts.stitch_ids import flag_clean, MIN_SEPARATION_PX
from scripts.stitch_assign import coexist_adj
from scripts.stitch_accumulate import find_seed
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)

UNFREEZE_BLOCKS = 2      # last N DINOv2 transformer blocks to fine-tune (few = less overfit on small data)
EPOCHS_ROUND    = 8      # epochs per accumulation round (backbone changes -> keep short)
BB_LR           = 1e-5   # SMALL backbone LR (protect pretrained weights)
HEAD_LR         = 1e-3   # larger head LR
BATCH           = 32
CONF_THRESH     = 0.90
MAX_ROUNDS      = 8

# augment train crops (safe for same-morph: geometry + brightness, NEVER hue/saturation) then the eval belt
_aug = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(8), T.ColorJitter(brightness=0.2)])
def train_tf(img):
    return transform(_aug(img))


def crop_paths(run_dir, frags):
    """(paths, crop_frag): every clean crop on disk, tagged with its fragment row index."""
    tracks = flag_clean(pd.read_parquet(os.path.join(run_dir, "tracks.parquet")), MIN_SEPARATION_PX)
    clean = tracks[tracks["clean"]]
    cdir = os.path.join(run_dir, "crops")
    paths, cf = [], []
    for i, fr in frags.iterrows():
        fid = int(fr["fish_id"])
        sub = clean[(clean["fish_id"] == fid) &
                    (clean["frame_number"] >= fr["frame_start"]) & (clean["frame_number"] <= fr["frame_end"])]
        for fn in sub["frame_number"]:
            p = os.path.join(cdir, f"fish_{fid}", f"frame_{int(fn)}_fish_{fid}.jpg")
            if os.path.exists(p):
                paths.append(p); cf.append(i)
    return paths, np.array(cf)


class CropDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def build_model(backbone_name, n_id, device):
    backbone = load_backbone(backbone_name, device)
    for p in backbone.parameters():
        p.requires_grad = False
    for blk in backbone.blocks[-UNFREEZE_BLOCKS:]:                  # unfreeze the top blocks
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True
    with torch.no_grad():                                          # infer feature dim
        feat_dim = backbone(torch.zeros(1, 3, 224, 224).to(device)).shape[1]
    head = nn.Linear(feat_dim, n_id).to(device)
    return backbone, head


def fine_tune(backbone, head, paths, crop_frag, labeled, device):
    idx, y = [], []
    for f, cls in labeled.items():
        ci = np.where(crop_frag == f)[0]
        idx.extend(ci.tolist()); y.extend([cls] * len(ci))
    dl = DataLoader(CropDS([paths[i] for i in idx], y, train_tf), batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam([{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": BB_LR},
                            {"params": head.parameters(), "lr": HEAD_LR}])
    lossf = nn.CrossEntropyLoss()
    backbone.train(); head.train()
    for _ in range(EPOCHS_ROUND):
        for imgs, lbl in dl:
            imgs, lbl = imgs.to(device), lbl.to(device)
            opt.zero_grad(); lossf(head(backbone(imgs)), lbl).backward(); opt.step()


def predict(backbone, head, paths, crop_frag, n_frags, device):
    backbone.eval(); head.eval()
    probs = {}
    with torch.no_grad():
        for f in range(n_frags):
            ci = np.where(crop_frag == f)[0]
            ps = []
            for j in range(0, len(ci), BATCH):
                imgs = torch.stack([transform(Image.open(paths[k]).convert("RGB")) for k in ci[j:j + BATCH]]).to(device)
                ps.append(torch.softmax(head(backbone(imgs)), dim=1).cpu())
            probs[f] = torch.cat(ps).mean(0).numpy()
    return probs


def main(video_name):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    run_dir = full["train_reid"]["videos"][video_name]["crops_run"]
    backbone_name = full["contrastive_reid"]["backbone"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    banner(f"STITCH — ACCUMULATION + BACKBONE FINE-TUNE ({video_name}, {device})")
    frags = pd.read_csv(os.path.join(run_dir, "stitch", "fragments.csv"))
    n_id = int(frags["fish_id"].nunique())
    frag_range = {i: (int(r["frame_start"]), int(r["frame_end"])) for i, r in frags.iterrows()}
    sizes = {i: int(r["n_frames"]) for i, r in frags.iterrows()}
    adj = coexist_adj(frag_range)
    paths, crop_frag = crop_paths(run_dir, frags)
    logger.info(f"{len(paths)} clean crops across {len(frags)} fragments")

    seed = find_seed(frag_range, sizes, n_id)
    labeled = {f: k for k, f in enumerate(seed)}
    logger.info(f"seed global fragment: {seed} -> identities {list(range(len(seed)))}")

    backbone, head = build_model(backbone_name, n_id, device)
    banner_sub("accumulating (fine-tune -> accept confident -> refine)")
    for rnd in range(1, MAX_ROUNDS + 1):
        fine_tune(backbone, head, paths, crop_frag, labeled, device)
        probs = predict(backbone, head, paths, crop_frag, len(frags), device)
        added = 0
        for f in sorted((f for f in range(len(frags)) if f not in labeled), key=lambda f: -probs[f].max()):
            cls, conf = int(probs[f].argmax()), float(probs[f].max())
            if conf < CONF_THRESH:
                break
            if any(labeled.get(g) == cls for g in adj[f]):
                continue
            labeled[f] = cls; added += 1
        logger.info(f"  round {rnd}: +{added}  ({len(labeled)}/{len(frags)} labelled)")
        if added == 0:
            break

    fine_tune(backbone, head, paths, crop_frag, labeled, device)
    probs = predict(backbone, head, paths, crop_frag, len(frags), device)
    for f in range(len(frags)):
        if f in labeled:
            continue
        forbidden = {labeled.get(g) for g in adj[f]}
        labeled[f] = next((int(c) for c in np.argsort(-probs[f]) if c not in forbidden), int(probs[f].argmax()))

    coll = sum(1 for f in adj for g in adj[f] if f < g and labeled[f] == labeled[g])
    banner("DONE")
    logger.info(f"all {len(frags)} fragments labelled | identities {len(set(labeled.values()))}/{n_id} | collisions {coll}")
    frags["cluster"] = [labeled[i] for i in range(len(frags))]
    out = os.path.join(run_dir, "stitch", "fragments_stitched.csv")
    frags.to_csv(out, index=False)
    logger.info(f"saved -> {out}")
    logger.info("VIEW the crossing:  python -m scripts.stitch_render --video_name %s --start_frame 840 --end_frame 870" % video_name)
    logger.info("WHOLE-video check:  python -m scripts.stitch_timeline --video_name %s" % video_name)


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Accumulation with backbone fine-tuning (idtracker.ai method)")
    parser.add_argument("--video_name", default=default_video)
    args = parser.parse_args()
    main(args.video_name)
