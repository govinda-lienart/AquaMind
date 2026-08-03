"""
stitch_assign.py — Stage 6 STITCHER piece 3.5: COEXISTENCE-CONSTRAINED identity assignment.

Plain k-means gave 29% frames with two fish sharing an ID — because it groups by looks alone and ignores
the obvious rule: two fragments that OVERLAP IN TIME must be different fish (can't be two places at once).
This replaces k-means with a greedy assignment that RESPECTS that rule:

  - process fragments longest-first (most reliable),
  - keep a running appearance prototype per identity,
  - assign each fragment to the most similar identity that is NOT already used by a fragment coexisting
    with it (forbidden set), opening a new identity when nothing matches well enough.

By construction, no two coexisting fragments get the same identity -> the frame collisions vanish. Uses
the trained contrastive head + cached frozen feats (no retrain). Writes fragments_stitched.csv (column
'cluster' = the constrained identity) for stitch_render to paint.

Usage:  python -m scripts.stitch_assign --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from scripts.contrastive_reid import Projection, embed_fragments
from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)

NEW_ID_THRESHOLD = 0.5   # if a fragment's best match to an existing identity is below this cosine, open a new identity


def coexist_adj(frag_range):
    """adjacency: frag -> set of fragments overlapping it in time (guaranteed different fish)."""
    ids = list(frag_range)
    adj = {f: set() for f in ids}
    for a in range(len(ids)):
        s1, e1 = frag_range[ids[a]]
        for b in range(a + 1, len(ids)):
            s2, e2 = frag_range[ids[b]]
            if s1 <= e2 and s2 <= e1:                    # ranges overlap
                adj[ids[a]].add(ids[b]); adj[ids[b]].add(ids[a])
    return adj


def constrained_assign(fps, sizes, frag_range, n_id):
    """Greedy longest-first assignment: best-matching identity that no coexisting fragment already holds."""
    adj = coexist_adj(frag_range)
    order = sorted(frag_range, key=lambda f: -sizes[f])   # longest (most reliable) first
    proto = np.zeros((n_id, fps.shape[1])); weight = np.zeros(n_id)
    assign = {}
    for f in order:
        forbidden = {assign[g] for g in adj[f] if g in assign}
        best_id, best_sim, empty = None, -2.0, None
        for k in range(n_id):
            if k in forbidden:
                continue
            if weight[k] == 0:
                empty = k if empty is None else empty     # remember one free slot
                continue
            sim = float(fps[f] @ (proto[k] / (np.linalg.norm(proto[k]) + 1e-8)))
            if sim > best_sim:
                best_sim, best_id = sim, k
        if best_id is None or (best_sim < NEW_ID_THRESHOLD and empty is not None):
            chosen = empty if empty is not None else best_id   # open a fresh identity
        else:
            chosen = best_id
        if chosen is None:                                # fully constrained fallback (shouldn't happen with n_id fish)
            chosen = next(k for k in range(n_id) if k not in forbidden)
        assign[f] = chosen
        proto[chosen] += fps[f] * sizes[f]; weight[chosen] += sizes[f]
    return assign


def collisions(assign, frag_range):
    """count coexisting fragment PAIRS given the same identity (0 = constraint satisfied)."""
    adj = coexist_adj(frag_range)
    return sum(1 for f in adj for g in adj[f] if f < g and assign[f] == assign[g])


def continuity_smooth(assign, frags, adj, min_hold=200, rounds=10):
    """Respect tracker continuity: a tracker track should keep ONE identity unless a swap is SUSTAINED.
    Flip a fragment to a neighbour's identity when it's (a) sandwiched between two fragments of the SAME
    other identity, or (b) a SHORT run (< min_hold frames) next to a differing neighbour — never if it
    would break the coexistence rule. Isolated appearance flips (like the frame-1807 error) collapse away;
    a long sustained block (a real silent swap) survives."""
    by_track = {}
    for i, r in frags.iterrows():
        by_track.setdefault(int(r["fish_id"]), []).append(i)
    for fid in by_track:
        by_track[fid].sort(key=lambda f: frags.loc[f, "frame_start"])
    size = {i: int(frags.loc[i, "n_frames"]) for i in frags.index}

    for _ in range(rounds):
        changed = False
        for seq in by_track.values():
            for i, f in enumerate(seq):
                cur = assign[f]
                left = assign[seq[i - 1]] if i > 0 else None
                right = assign[seq[i + 1]] if i < len(seq) - 1 else None
                target = None
                if left is not None and left == right and cur != left:          # sandwiched X-Y-X -> X
                    target = left
                elif size[f] < min_hold:                                        # short run next to a stable neighbour
                    if left is not None and cur != left and (right is None or right == left):
                        target = left
                    elif right is not None and cur != right and (left is None or left == right):
                        target = right
                if target is not None and target != cur and all(assign[g] != target for g in adj[f]):
                    assign[f] = target; changed = True
        if not changed:
            break
    return assign


def main(video_name):
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    run_dir = full["train_reid"]["videos"][video_name]["crops_run"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    banner(f"STITCH piece 3.5 — coexistence-constrained assignment ({video_name})")
    frags = pd.read_csv(os.path.join(run_dir, "stitch", "fragments.csv"))
    n_id = int(frags["fish_id"].nunique())
    frag_range = {i: (int(r["frame_start"]), int(r["frame_end"])) for i, r in frags.iterrows()}
    sizes = {i: int(r["n_frames"]) for i, r in frags.iterrows()}

    # per-fragment contrastive embeddings (reuse cached feats + trained head — no retrain)
    z = np.load(os.path.join(run_dir, "stitch", "contrastive_cache.npz"))
    feats, crop_frag = torch.from_numpy(z["feats"]).float(), z["crop_frag"]
    ck = torch.load(os.path.join(run_dir, "stitch", "contrastive_head.pt"), map_location="cpu")
    head = Projection(ck["in_dim"], ck["hidden_dim"], ck["out_dim"]); head.load_state_dict(ck["head_state"]); head.to(device)
    fps = embed_fragments(head, feats, crop_frag, len(frags), device)

    banner_sub("compare: naive k-means vs coexistence-constrained")
    km = KMeans(n_clusters=n_id, n_init=10, random_state=0).fit_predict(fps)
    km_assign = {i: int(km[i]) for i in range(len(frags))}
    logger.info(f"k-means             : coexisting same-id fragment pairs = {collisions(km_assign, frag_range)}  (the bug)")

    assign = constrained_assign(fps, sizes, frag_range, n_id)
    logger.info(f"constrained assign  : {collisions(assign, frag_range)} collisions | "
                f"{sum(1 for f in assign)} frags | id-changes/track before smoothing")

    adj = coexist_adj(frag_range)
    n_before = sum(1 for f in range(len(frags)) for g in adj[f] if f < g and assign[f] == assign[g])
    assign = continuity_smooth(assign, frags, adj)
    n_after = sum(1 for f in range(len(frags)) for g in adj[f] if f < g and assign[f] == assign[g])
    logger.info(f"continuity-smoothed : coexisting same-id fragment pairs {n_before} -> {n_after}  (isolated flips collapsed to track identity)")
    logger.info(f"identities used: {len(set(assign.values()))}/{n_id}")

    frags["cluster"] = [assign[i] for i in range(len(frags))]   # 'cluster' col so stitch_render paints it unchanged
    out = os.path.join(run_dir, "stitch", "fragments_stitched.csv")
    frags.to_csv(out, index=False)
    banner("DONE")
    logger.info(f"saved -> {out}")
    logger.info("now: python -m scripts.stitch_render --video_name %s  (it will use fragments_stitched.csv)" % video_name)


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Coexistence-constrained identity assignment")
    parser.add_argument("--video_name", default=default_video)
    args = parser.parse_args()
    main(args.video_name)
