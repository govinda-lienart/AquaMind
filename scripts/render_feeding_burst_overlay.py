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
from scipy.signal import find_peaks

from scripts.video_utils import grab_video_name
from scripts.console import banner, banner_sub

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS

VIDEO_RUN_NAME = "IMG_2349_appearance_2026_08_12_1926"
FOOD_START = 14491
SMOOTH_WIN = 5 # rolling_avg width (frames) to kill single frame tracker jitter wihtout blurring real burst # choosen arbitrary but typically safe value to start with.
BURST_PROM = 3.0   # chosen  arbitrary-but-reasonable guess-  cm/s — how far a peak must stand out above its surrounding baseline to count (not just barely poking over BURST_MIN_CMS)
BURST_GAP_S = 1.0  #  seconds —> choosen arbitraly minimum time between two peaks for the same fish to count as separate bursts, not one burst counted twice


# ── STEP 1 — load tracks + resolve the video path ────────────────────────────
parquet_path, pixels_per_cm, *_ = grab_video_name(VIDEO_RUN_NAME)
tracks = pd.read_parquet(parquet_path)
run_dir = os.path.dirname(parquet_path)
video_path = os.path.join(run_dir, f"tracker_{VIDEO_RUN_NAME}.mp4")

# ── STEP 1b — per-fish smoothed speed + data-derived BURST_MIN_CMS ────────────


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
banner("STEP 2 — per-fish burst detection")

FISH_IDS = sorted(tracks.fish_id.unique()) # sorted is python built - gives small list [1, 2, 3, 4]. 
burst_frames_by_fish = {}  # burst_frames_by_fish is a dictionary where each key is a fish_id, and each value is an array of frame numbers where that fish had a burst. e.g {1: [820, 1450, 8901, ...], 2: [340, 5210, ...], 3: [...

banner_sub("find bursts per fish (height + prominence + distance gates)")

burst_frames_by_fish = {}
for fid in FISH_IDS:
    g = tracks[tracks.fish_id == fid].sort_values("frame_number")
    median_dt = g["timestamp"].diff().mean() # find peaks - distance needs only one value argument,....so we take the median 
    gap_frames = max(1, int(round(BURST_GAP_S / median_dt)))
    peak_idx, _ = find_peaks( # imported module function # peak_idx is an array an array of position numbers # Each individual peak found by find_peaks = one single frame - the exact row where that burst's speed hit its summit (its single highest point). 
        g["speed_smooth"].to_numpy(),
        height=BURST_MIN_CMS,      # filter 1: fast enough in cm/s, full stop?
        prominence=BURST_PROM,     # filter 2: does it stand out from its own dip, or is it just a small hill riding on a bigger one?
        distance=gap_frames,       # filter 3: not too close in time to an already-counted, taller peak?
    )  
    # gures out which real frame numbers this fish's bursts happened on, and stores them under that fish's ID.
    frame_numbers_array = g["frame_number"].to_numpy()   # step 1: whole frame_number column, as an array
    burst_frames = frame_numbers_array[peak_idx]          # step 2: just the rows where a burst peaked
    burst_frames_by_fish[fid] = burst_frames               # step 3: file it under this fish's id - > burst_frames_by_fish is a dictionary
    logging.info(f"fish {fid}: {len(burst_frames)} bursts detected")

logging.info(f"fish 1 bursts: {burst_frames_by_fish[1]}")

# ── STEP 2b — segment label + flat burst-events table ────────────────────────
banner("STEP 2b — segment label + flat burst-events table")

banner_sub("build one flat table: every burst, with its timestamp + segment")
rows = []
for fid, frame_numbers in burst_frames_by_fish.items():
    for frame_number in frame_numbers:
        timestamp = tracks.loc[
            (tracks.fish_id == fid) & (tracks.frame_number == frame_number), "timestamp"
        ].iloc[0]  # .iloc[0] -> there's exactly one row for this (fish_id, frame_number) pair, grab its value
        segment = "control" if frame_number < FOOD_START else "feeding"
        rows.append({"fish_id": fid, "frame_number": frame_number, "timestamp": timestamp, "segment": segment})

burst_events = pd.DataFrame(rows, columns=["fish_id", "frame_number", "timestamp", "segment"])
logging.info(burst_events.to_string())

banner_sub("helpers built on top of burst_events")

def bursts_up_to(fish_id, frame_number, segment):
    # running count for this fish, WITHIN this segment only — a control-segment
    # count naturally stops growing once frame_number crosses into feeding,
    # since there are no more control-segment rows left to count
    sub = burst_events[(burst_events.fish_id == fish_id) & (burst_events.segment == segment)]
    return int((sub.frame_number <= frame_number).sum())

def is_flickering(fish_id, frame_number):
    # True only on the exact frame a burst peaked — used to flash the overlay
    return bool(((burst_events.fish_id == fish_id) & (burst_events.frame_number == frame_number)).any())


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
