"""
render_feeding_burst_overlay.py — live per-fish burst overlay on the tracked video.

usage:  python -m scripts.render_feeding_burst_overlay
"""

# ── imports ──────────────────────────────────────────────────────────────────
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.video_utils import grab_video_name
from scripts.console import banner, banner_sub

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS

VIDEO_RUN_NAME = "IMG_2349_appearance_2026_08_12_1926"
FOOD_START = 14491
SMOOTH_WIN = 5 # rolling_avg width (frames) to kill single frame tracker jitter wihtout blurring real burst # choosen arbitrary but typically safe value to start with.


# ── STEP 1 — load tracks + resolve the video path ────────────────────────────
parquet_path, pixels_per_cm, *_ = grab_video_name(VIDEO_RUN_NAME)
tracks = pd.read_parquet(parquet_path)
run_dir = os.path.dirname(parquet_path)
video_path = os.path.join(run_dir, f"tracker_{VIDEO_RUN_NAME}.mp4")

# ── STEP 1b — per-fish smoothed speed + data-derived BURST_MIN_CMS ────────────

FISH_IDS = sorted(tracks.fish_id.unique()) # sorted is python built - gives small list [1, 2, 3, 4]. 

banner("STEP 1b — per-fish smoothed speed + data-derived BURST_MIN_CMS")

# compute per-fish distance + speed
banner_sub("compute per-fish distance + speed")
tracks = tracks.sort_values(["fish_id", "frame_number"])
dx = tracks.groupby("fish_id")["x"].diff()
dy = tracks.groupby("fish_id")["y"].diff()
dist_px = np.sqrt(dx**2 + dy**2) # pythagoras - distance (pixels)
dist_cm = dist_px / pixels_per_cm  # series
tracks["dist_px"] = dist_px
tracks["dist_cm"] = dist_cm
dt = tracks.groupby("fish_id")["timestamp"].diff() # delta time - difference betrween rows
tracks["speed_cms"] = dist_cm / dt
grouped_speed = tracks.groupby("fish_id")["speed_cms"]
tracks["speed_smooth"] = grouped_speed.transform(lambda s: s.rolling(SMOOTH_WIN, min_periods=1).mean()) # rolling creates a window...that moves from left to right...here of 5...and min_period..is small window for the initial values...and the mean takes the average for each window
logging.info(tracks[["fish_id", "frame_number", "dist_px", "dist_cm", "speed_cms", "speed_smooth"]].head().to_string())

# derive BURST_MIN_CMS from control-segment speed
banner_sub("derive BURST_MIN_CMS from control-segment speed")
control_speeds = tracks.loc[tracks.frame_number < FOOD_START, "speed_smooth"] # filter .loc[row_condition, column_name] 
BURST_MIN_CMS= control_speeds.quantile(0.95) #  95th percentile ...is a fraction..rthe vlaue below whcih 95 percent of controls speeds fall cms = centimeters per second (cm/s)
logging.info(f"BURST_MIN_CMS = {BURST_MIN_CMS:.2f} cm/s (95th percentile of control-segment speed)")
feeding_speeds = tracks.loc[tracks.frame_number >= FOOD_START, "speed_smooth"]

# graph  BURST_MIN_CMS from control-segment speed
feeding_burst_output = os.path.join(run_dir, "feeding_burst")
os.makedirs(feeding_burst_output, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 40, 101) # 
ax.hist(control_speeds.dropna(), bins=bins, alpha=0.6, label="control", color="tab:blue", density=True)
ax.hist(feeding_speeds.dropna(), bins=bins, alpha=0.6, label="feeding", color="tab:orange", density=True)
ax.axvline(BURST_MIN_CMS, color="red", linestyle="--", label=f"BURST_MIN_CMS = {BURST_MIN_CMS:.2f} cm/s")
ax.set_xlabel("speed_smooth (cm/s)")
ax.set_ylabel("density")
ax.set_title("Control vs feeding speed distribution (threshold from control only)")
ax.set_xlim(0, 40)
ax.legend()
hist_path = os.path.join(feeding_burst_output, "burst_threshold_histogram.png")
fig.savefig(hist_path, dpi=150)
logging.info(f"saved evidence figure -> {hist_path}")

# ── STEP 2 — per-fish burst detection ─────────────────────────────────────────
# find_peaks has no groupby equivalent, so this is the one place a real per-fish
# loop is unavoidable: for fid in FISH_IDS: g = tracks[tracks.fish_id == fid]
# (tracks already has speed_smooth from STEP 1b, no recomputation needed)
# using g["speed_smooth"] + BURST_MIN_CMS (STEP 1b) + BURST_PROM + BURST_GAP_S
# fill burst_frames_by_fish: {fish_id: sorted array of frame numbers where a burst peaked}
# (this dict is fine/expected here — it's peak-detection RESULTS, not a second copy
# of the tracking data itself, so it doesn't reintroduce the earlier confusion)
# (this part stays ONE running total per fish here — the control/feeding split
# happens next, in STEP 2b, not inside this loop)



# ── STEP 2b — segment label + flat burst-events table ────────────────────────
# for every (fish_id, frame_number) burst found above, look up:
#   - timestamp        -> tracks.loc[tracks.frame_number == frame_number, "timestamp"]
#   - segment           -> "control" if frame_number < FOOD_START else "feeding"
# stack all fish into one flat table:
#   burst_events = pd.DataFrame(rows, columns=["fish_id", "frame_number", "timestamp", "segment"])
# this table is the single source of truth for both the on-screen counters (STEP 3)
# and the saved data + graph (STEP 4/5) — build it once, reuse it everywhere

# helper: bursts_up_to(fish_id, frame_number, segment) -> running count for that fish
#         WITHIN that segment only (filter burst_events to fish_id + segment, then
#         count how many frame_number <= the current frame — same searchsorted idea
#         as before, just on the filtered array instead of the whole-video one)
# helper: is_flickering(fish_id, frame_number) -> True if a burst just peaked
#         (unchanged — doesn't need the segment split, a flicker is a flicker)


# ── STEP 3 — render the overlay video ────────────────────────────────────────
# open the tracked video, loop frames, per fish per frame:
#   control_count = bursts_up_to(fid, frame_number, "control")   # freezes once feeding starts
#   feeding_count = bursts_up_to(fid, frame_number, "feeding")   # stays 0 until FOOD_START
#   draw f"F{fid}  control:{control_count}  feeding:{feeding_count}" + star marker if flickering
# write output the same way as a single-counter overlay would


# ── STEP 4 — save burst_events to disk ────────────────────────────────────────
# write burst_events (STEP 2b) to output/burst_events_<RUN_STAMP>.parquet (or .csv) —
# this is the file the graph step (and any later re-analysis) reads back, so the
# render loop above never has to be re-run just to change how the graph looks


# ── STEP 5 — graph: burst rate, control vs feeding ────────────────────────────
# read burst_events back, bin by e.g. 10s windows (RATE_BIN_S), count bursts per bin
# (pooled across all fish -> identity-independent, like the earlier salvage metric)
# plot the rate-over-time curve (matplotlib, Agg backend), then shade the control/
# feeding spans and draw a dashed horizontal line at EACH segment's mean rate, labeled
# e.g. "control\nmean 2.9/s" / "feeding\nmean 4.1/s" -> same technique as the earlier
# burst_rate_over_time.png reference plot (phase mean lines), just 2 segments not 3
# save PNG -> this is the actual evidence figure for "did burst rate change after
# food went in", not just the on-screen counters
