"""
stitch_timeline.py — visualise the stitcher's identity assignment across the WHOLE video.

One lane per tracker fish_id, time on the x-axis, colour = stitched identity. Read it:
  - each lane mostly ONE colour  -> the stitcher held that track's identity steadily
  - a vertical slice shows 5 DIFFERENT colours -> no collision at that moment (good)
  - same colour in two lanes at the same x -> COLLISION (two fish, one identity) = an error
  - a lane flipping colours mid-open-water -> a suspicious re-assignment

Usage:  python -m scripts.stitch_timeline --video_name IMG_1839
"""
import os
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scripts.console import banner
from scripts.logger import setup_logging

logger = logging.getLogger(__name__)
PALETTE = ["#e05050", "#5aa0ff", "#50c850", "#f0a000", "#c050ff"]   # identities 1..5
GREY = "#c8c8c8"


def runs(frames, clusters):
    """contiguous (start, end, cluster) runs along a track's timeline."""
    out = []
    s = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or clusters[i] != clusters[s] or frames[i] != frames[i - 1] + 1:
            out.append((frames[s], frames[i - 1], clusters[s])); s = i
    return out


def main(video_name):
    with open("config.yaml") as f:
        run_dir = yaml.safe_load(f)["train_reid"]["videos"][video_name]["crops_run"]
    banner(f"STITCH timeline — identity across the whole video ({video_name})")
    t = pd.read_parquet(os.path.join(run_dir, "stitch", "tracks_stitched.parquet"))
    fishids = sorted(t["fish_id"].unique())

    # collision count per frame (same stitched id on 2+ fish) — the honest error signal
    g = t[t["cluster"] >= 0].groupby(["frame_number", "cluster"]).size()
    coll_frames = g[g > 1].index.get_level_values("frame_number").unique()
    lo, hi = int(t["frame_number"].min()), int(t["frame_number"].max())

    fig, ax = plt.subplots(figsize=(16, 5))
    for lane, fid in enumerate(fishids):
        gg = t[t["fish_id"] == fid].sort_values("frame_number")
        fr = gg["frame_number"].to_numpy(); cl = gg["cluster"].to_numpy()
        for s, e, c in runs(fr, cl):
            color = GREY if c < 0 else PALETTE[int(c) % 5]
            ax.broken_barh([(s, e - s + 1)], (lane * 10, 8), facecolors=color)
    # mark collision frames as a thin red rug along the top
    ax.broken_barh([(f, 1) for f in coll_frames], (len(fishids) * 10 + 2, 4), facecolors="black")

    ax.set_yticks([lane * 10 + 4 for lane in range(len(fishids))])
    ax.set_yticklabels([f"tracker Fish {fid}" for fid in fishids])
    ax.set_xlabel("frame (time →)"); ax.set_xlim(lo, hi)
    ax.set_title(f"stitched identity across the video  (colour = identity; black rug = collision frames: "
                 f"{len(coll_frames)}/{t['frame_number'].nunique()} = {100*len(coll_frames)/t['frame_number'].nunique():.0f}%)")
    legend = [Patch(facecolor=PALETTE[k], label=f"identity {k+1}") for k in range(5)] + \
             [Patch(facecolor=GREY, label="crossing (no id)"), Patch(facecolor="black", label="collision frame")]
    ax.legend(handles=legend, ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    out = os.path.join(run_dir, "stitch", "stitch_timeline.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"saved -> {out}")
    logger.info("read it: each lane one colour = steady; same colour in 2 lanes at same x = collision (error).")


if __name__ == "__main__":
    setup_logging()
    with open("config.yaml") as f:
        default_video = yaml.safe_load(f)["contrastive_reid"]["video"]
    parser = argparse.ArgumentParser(description="Timeline of stitched identity across the video")
    parser.add_argument("--video_name", default=default_video)
    args = parser.parse_args()
    main(args.video_name)
