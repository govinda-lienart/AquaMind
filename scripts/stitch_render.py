"""
stitch_render.py — Stage 6 STITCHER piece 4: relabel tracks by cluster + render BEFORE/AFTER video.

The decisive, label-INDEPENDENT proof: take the contrastive clustering (fragments_contrastive.csv, one
cluster per fragment = the stitcher's identity answer), stamp it back onto every track, and render a
SIDE-BY-SIDE video:
    LEFT  = BEFORE (tracker's original fish_id — swaps at crossings)
    RIGHT = AFTER  (stitched cluster id — should stay glued to the same physical fish)
Watch the right half: if a fish's colour/id holds through a crossing where the left half swaps, the
stitcher worked. This is both the proof AND the portfolio artifact.

Relabel logic: each clean fragment owns [frame_start, frame_end] for its tracker fish_id -> stamp its
cluster there. Crossing/ghost frames (not in any fragment) are FORWARD/BACK-FILLED within each tracker
track (carry the last confident cluster across the gap) — the cheap version of idtracker's crossing
back-fill. Boxes are FIXED-SIZE (parquet has no bbox), centred on (x,y).

Usage:  python -m scripts.stitch_render --video_name IMG_1839 [--start_frame N --end_frame M]
"""
import os
import argparse
import logging

import yaml
import cv2
import numpy as np
import pandas as pd

from scripts.console import banner, banner_sub
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)

BOX_W, BOX_H = 130, 90                                    # fixed crop-ish box (parquet has no bbox)
PALETTE = [(255, 80, 80), (80, 180, 255), (80, 255, 80), (0, 220, 255), (255, 0, 255)]  # BGR, ids 1..5
GREY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def relabel(tracks, frags):
    """Add a 'cluster' column to tracks: each fragment stamps its cluster onto its frames; crossing/ghost
    gaps are filled by carrying the last confident cluster within each tracker track."""
    tracks = tracks.copy()
    tracks["cluster"] = -1
    for _, fr in frags.iterrows():                        # stamp each clean fragment's cluster
        m = ((tracks["fish_id"] == fr["fish_id"]) &
             (tracks["frame_number"] >= fr["frame_start"]) & (tracks["frame_number"] <= fr["frame_end"]))
        tracks.loc[m, "cluster"] = fr["cluster"]
    tracks = tracks.sort_values(["fish_id", "frame_number"])
    tracks["cluster"] = (tracks.groupby("fish_id")["cluster"]      # fill crossing gaps within each track
                         .transform(lambda s: s.replace(-1, np.nan).ffill().bfill()))
    tracks["cluster"] = tracks["cluster"].fillna(-1).astype(int)
    return tracks


def draw_box(img, x, y, color, label):
    x1, y1 = int(x - BOX_W / 2), int(y - BOX_H / 2)
    x2, y2 = int(x + BOX_W / 2), int(y + BOX_H / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 6), FONT, 0.6, color, 2)


def panel(frame, rows, key, title):
    """Draw one half: boxes coloured by `key` ('fish_id' before, or 'cluster' after)."""
    img = frame.copy()
    for r in rows:
        if key == "fish_id":
            cid = int(r["fish_id"]); color = PALETTE[(cid - 1) % 5]; label = f"Fish {cid}"
        else:
            c = int(r["cluster"]); color = GREY if c < 0 else PALETTE[c % 5]; label = "?" if c < 0 else f"Fish {c + 1}"
        draw_box(img, r["x"], r["y"], color, label)
    cv2.putText(img, title, (20, 40), FONT, 1.0, (255, 255, 255), 2)
    return img


def main(video_name, start_frame, end_frame):
    with open("config.yaml") as f:
        run_dir = yaml.safe_load(f)["train_reid"]["videos"][video_name]["crops_run"]
    video_path = f"videos/{video_name}.MOV"

    banner(f"STITCH piece 4 — relabel + render BEFORE/AFTER ({video_name})")
    tracks = pd.read_parquet(os.path.join(run_dir, "tracks.parquet"))
    stitched = os.path.join(run_dir, "stitch", "fragments_stitched.csv")     # constrained (piece 3.5) if present
    frags_file = stitched if os.path.exists(stitched) else os.path.join(run_dir, "stitch", "fragments_contrastive.csv")
    frags = pd.read_csv(frags_file)
    logger.info(f"{len(tracks)} track rows | {len(frags)} fragments | {frags['cluster'].nunique()} identities | from {os.path.basename(frags_file)}")

    tracks = relabel(tracks, frags)
    tracks.to_parquet(os.path.join(run_dir, "stitch", "tracks_stitched.parquet"))
    n_swapped = int((tracks["fish_id"] - 1 != tracks["cluster"]).sum())    # rows where stitched id != tracker id (informative only, tracker id is arbitrary)
    logger.info(f"relabelled — saved tracks_stitched.parquet ({n_swapped} rows differ from tracker id)")

    by_frame = {fn: g.to_dict("records") for fn, g in tracks.groupby("frame_number")}
    lo = start_frame if start_frame is not None else int(tracks["frame_number"].min())
    hi = end_frame if end_frame is not None else int(tracks["frame_number"].max())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w, half_h = 960, int(960 * H / W)                 # each panel downscaled; concat side-by-side
    out_path = os.path.join(run_dir, "stitch", f"stitched_before_after_{lo}_{hi}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (half_w * 2, half_h))

    banner_sub(f"rendering frames {lo}–{hi}  (LEFT tracker | RIGHT stitched)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    written = 0
    for fn in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        rows = by_frame.get(fn, [])
        before = cv2.resize(panel(frame, rows, "fish_id", "BEFORE (tracker)"), (half_w, half_h))
        after = cv2.resize(panel(frame, rows, "cluster", "AFTER (stitched)"), (half_w, half_h))
        combo = np.hstack([before, after])
        cv2.line(combo, (half_w, 0), (half_w, half_h), (255, 255, 255), 2)
        writer.write(combo)
        written += 1
        if written % 1000 == 0:
            logger.info(f"  {written} frames written…")
    cap.release(); writer.release()
    banner("DONE")
    logger.info(f"wrote {written} frames -> {out_path}")
    logger.info("WATCH the right half: does each fish's id hold through a crossing where the LEFT half swaps? "
                "That is the label-independent proof.")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Stitcher piece 4: relabel tracks + render before/after video")
    parser.add_argument("--video_name", default=default_video)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    args = parser.parse_args()
    main(args.video_name, args.start_frame, args.end_frame)
