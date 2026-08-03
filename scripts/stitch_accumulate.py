"""
stitch_accumulate.py — Stage 6 STITCHER: the ACCUMULATION protocol (idtracker.ai / TRex core trick).

Our contrastive stitcher failed because its features DRIFT over the 2-min video (a fish looks different at
frame 500 vs 7000). Accumulation fixes drift by SELF-TRAINING a recognizer on progressively more diverse,
CONFIDENT data until it recognizes each fish everywhere:

  1. SEED: find a "global fragment" — a frame where all N fish are simultaneously visible AND separated
     -> those N fragments are guaranteed different fish -> label them identities 0..N-1.
  2. TRAIN a classifier head on the labelled crops (frozen DINOv2 feats -> Linear -> N classes).
  3. PREDICT every unlabelled fragment; ACCEPT the ones the head is CONFIDENT about (mean softmax >= thresh)
     AND that don't clash with a coexisting labelled fragment (can't be two places at once).
  4. RETRAIN on seed + accepted (now more diverse — spans more of the video) and repeat until nothing new
     can be confidently added. The head learns cross-time invariance BECAUSE it's trained on both ends.
  5. Assign the leftover low-confidence fragments by the final head, respecting coexistence.

Uses the cached FROZEN DINOv2 features (contrastive_cache.npz) — no backbone re-run. Writes
fragments_stitched.csv (column 'cluster' = accumulated identity) for stitch_render / stitch_timeline.

Usage:  python -m scripts.stitch_accumulate --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.stitch_assign import coexist_adj
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)

CONF_THRESH = 0.90   # accept a fragment only if the head's mean softmax confidence >= this
EPOCHS      = 60     # head training epochs per round (frozen feats -> fast)
MAX_ROUNDS  = 12


def find_seed(frag_range, sizes, n_id):
    """A global fragment: N mutually-coexisting fragments (a frame where all N fish are separated).
    Returns the highest-total-length such set, or the max-coexisting set if N is never all-visible."""
    best, best_score, best_n = None, -1, 0
    for t in sorted({s for s, _ in frag_range.values()} | {e for _, e in frag_range.values()}):
        active = [f for f, (s, e) in frag_range.items() if s <= t <= e]
        score = sum(sizes[f] for f in active)
        if len(active) == n_id and score > best_score:
            best, best_score = active, score
        if len(active) > best_n:                                  # fallback tracker
            best_n, fallback = len(active), active
    return best if best is not None else fallback


def train_head(feats, frag_to_crops, labeled, feat_dim, n_id, device, epochs):
    idx, y = [], []
    for f, cls in labeled.items():
        idx.extend(frag_to_crops[f].tolist()); y.extend([cls] * len(frag_to_crops[f]))
    X = feats[idx].to(device); Y = torch.tensor(y, device=device)
    head = nn.Linear(feat_dim, n_id).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(head(X), Y).backward(); opt.step()
    head.eval()
    return head


def predict(head, feats, frag_to_crops, device):
    """Per-fragment averaged softmax -> (full prob vector)."""
    probs = {}
    with torch.no_grad():
        for f, ci in frag_to_crops.items():
            probs[f] = torch.softmax(head(feats[ci].to(device)), dim=1).mean(0).cpu().numpy()
    return probs


def main(video_name):
    with open("config.yaml") as f:
        run_dir = yaml.safe_load(f)["train_reid"]["videos"][video_name]["crops_run"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    banner(f"STITCH — ACCUMULATION protocol ({video_name})")
    frags = pd.read_csv(os.path.join(run_dir, "stitch", "fragments.csv"))
    n_id = int(frags["fish_id"].nunique())
    frag_range = {i: (int(r["frame_start"]), int(r["frame_end"])) for i, r in frags.iterrows()}
    sizes = {i: int(r["n_frames"]) for i, r in frags.iterrows()}
    adj = coexist_adj(frag_range)

    z = np.load(os.path.join(run_dir, "stitch", "contrastive_cache.npz"))       # FROZEN DINOv2 feats + crop->fragment
    feats = torch.from_numpy(z["feats"]).float()
    crop_frag = z["crop_frag"]
    frag_to_crops = {f: np.where(crop_frag == f)[0] for f in range(len(frags))}
    feat_dim = feats.shape[1]

    # ---------- 1. seed from a global fragment ----------
    seed = find_seed(frag_range, sizes, n_id)
    labeled = {f: k for k, f in enumerate(seed)}
    logger.info(f"seed (all-{n_id}-visible global fragment): fragments {seed} -> identities {list(range(len(seed)))}")

    # ---------- 2-4. accumulate ----------
    banner_sub("accumulating (train -> accept confident -> retrain)")
    for rnd in range(1, MAX_ROUNDS + 1):
        head = train_head(feats, frag_to_crops, labeled, feat_dim, n_id, device, EPOCHS)
        probs = predict(head, feats, frag_to_crops, device)
        cands = sorted((f for f in range(len(frags)) if f not in labeled),
                       key=lambda f: -probs[f].max())                 # most-confident first
        added = 0
        for f in cands:
            cls, conf = int(probs[f].argmax()), float(probs[f].max())
            if conf < CONF_THRESH:
                break                                                 # rest are even less confident
            if any(labeled.get(g) == cls for g in adj[f]):            # coexistence: clash with a labelled coexisting fragment
                continue
            labeled[f] = cls; added += 1
        logger.info(f"  round {rnd}: +{added} fragments  ({len(labeled)}/{len(frags)} labelled)")
        if added == 0:
            break

    # ---------- 5. assign leftovers by the final head (respect coexistence) ----------
    head = train_head(feats, frag_to_crops, labeled, feat_dim, n_id, device, EPOCHS)
    probs = predict(head, feats, frag_to_crops, device)
    for f in range(len(frags)):
        if f in labeled:
            continue
        order = np.argsort(-probs[f])                                 # try classes best-first
        forbidden = {labeled.get(g) for g in adj[f]}
        labeled[f] = next((int(c) for c in order if c not in forbidden), int(order[0]))

    # ---------- report + save ----------
    coll = sum(1 for f in adj for g in adj[f] if f < g and labeled[f] == labeled[g])
    banner("DONE")
    logger.info(f"all {len(frags)} fragments labelled | identities used {len(set(labeled.values()))}/{n_id} | "
                f"coexisting same-id collisions {coll}")
    frags["cluster"] = [labeled[i] for i in range(len(frags))]
    out = os.path.join(run_dir, "stitch", "fragments_stitched.csv")
    frags.to_csv(out, index=False)
    logger.info(f"saved -> {out}")
    logger.info("now re-render / re-timeline: python -m scripts.stitch_render --video_name %s ; "
                "python -m scripts.stitch_timeline --video_name %s" % (video_name, video_name))


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Accumulation protocol: self-train a video-wide fish recognizer")
    parser.add_argument("--video_name", default=default_video)
    args = parser.parse_args()
    main(args.video_name)
