"""
render_feeding_burst_overlay.py — live per-fish burst overlay on the tracked video.

usage:  python -m scripts.render_feeding_burst_overlay
"""

# ── imports ──────────────────────────────────────────────────────────────────
import os

import cv2
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
BURST_PROM = 2.0   # lowered from 3.0 — catch smaller real feeding lunges that don't stand out as much from an already-elevated baseline, without touching BURST_MIN_CMS (applies to both segments, same find_peaks call)
BURST_GAP_S = 1.0  #  seconds —> choosen arbitraly minimum time between two peaks for the same fish to count as separate bursts, not one burst counted twice
RATE_BIN_S = 10.0  # seconds per bin for the STEP 5 rate curve — arbitrary but reasonable starting value
OVERLAY_FRAME_START = 0  # from here to the end of the video (video has 21678 frames total)
OVERLAY_FRAME_END = None     # None -> no end limit, runs until cap.read() reports no more frames


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

FLICKER_S = 0.35  # seconds the star marker stays visible after a burst peaks — long enough to actually see it


# ── STEP 3 — render the overlay video ────────────────────────────────────────
banner("STEP 3 — render the overlay video")

overlay_path = os.path.join(feeding_burst_output, f"burst_overlay_{VIDEO_RUN_NAME}.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_width, frame_height))  # avc1 (H.264) so QuickTime can actually open it — mp4v writes a valid file, just one QuickTime refuses

# fish x/y per frame, for drawing the burst marker AT the fish instead of just in a corner
fish_positions = tracks.set_index(["fish_id", "frame_number"])[["x", "y"]]
flicker_frames = max(1, int(FLICKER_S * fps))  # FLICKER_S converted to a frame count, this fish's video's own fps

# ── precompute frames_since_burst for every (fish_id, frame_number) we'll render ──
# merge_asof = a merge with "nearest match at-or-before" instead of an exact-key
# match, so this replaces the old np.searchsorted helper (same idea, pandas
# vocabulary) — and it's computed ONCE for the whole render range instead of once
# per fish per frame inside the loop.
banner_sub("precompute last-burst lookup via merge_asof")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
last_frame = OVERLAY_FRAME_END if OVERLAY_FRAME_END is not None else total_frames - 1
render_frames = np.arange(OVERLAY_FRAME_START, last_frame + 1)

# every (fish_id, frame_number) pair we'll actually render — the "left" side of the merge
# merge_asof needs the "on" column sorted GLOBALLY (not just within each fish_id group,
# which is what by="fish_id" handles separately) — sort by frame_number alone
lookup = pd.MultiIndex.from_product([FISH_IDS, render_frames], names=["fish_id", "frame_number"]).to_frame(index=False)
lookup = lookup.sort_values("frame_number")

burst_only = burst_events[["fish_id", "frame_number"]].sort_values("frame_number").rename(
    columns={"frame_number": "last_burst_frame"}
)

lookup = pd.merge_asof(
    lookup, burst_only,
    left_on="frame_number", right_on="last_burst_frame",
    by="fish_id", direction="backward",  # "backward" = nearest last_burst_frame <= frame_number
)
lookup["frames_since_burst"] = lookup["frame_number"] - lookup["last_burst_frame"]
lookup = lookup.set_index(["fish_id", "frame_number"])["frames_since_burst"]

cap.set(cv2.CAP_PROP_POS_FRAMES, OVERLAY_FRAME_START)  # jump straight there instead of decoding every earlier frame
frame_number = OVERLAY_FRAME_START
while True:
    if OVERLAY_FRAME_END is not None and frame_number > OVERLAY_FRAME_END:
        break
    ok, frame = cap.read()
    if not ok:  # ok is False once there are no more frames left to read
        break
    for row, fid in enumerate(FISH_IDS):
        control_count = bursts_up_to(fid, frame_number, "control")  # freezes once feeding starts
        feeding_count = bursts_up_to(fid, frame_number, "feeding")  # stays 0 until FOOD_START
        flickering = is_flickering(fid, frame_number)
        text = f"F{fid}  control:{control_count}  feeding:{feeding_count}" + (" *" if flickering else "")
        color = (0, 0, 255) if flickering else (255, 255, 255)  # flash red on the exact burst frame
        cv2.putText(frame, text, (10, 30 + row * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # burst marker, drawn right on the fish, visible for flicker_frames after it peaks
        diff = lookup.at[(fid, frame_number)]
        if pd.notna(diff) and 0 <= diff <= flicker_frames:
            try:
                x, y = fish_positions.loc[(fid, frame_number)]
            except KeyError:
                continue  # this fish has no detection on this exact frame (occluded)
            cv2.drawMarker(frame, (int(x), int(y) - 15), (0, 255, 255),
                            markerType=cv2.MARKER_STAR, markerSize=26, thickness=2)
    writer.write(frame)
    frame_number += 1

cap.release()
writer.release()
logging.info(f"saved overlay video -> {overlay_path}")


# ── STEP 4 — save burst_events to disk ────────────────────────────────────────
banner("STEP 4 — save burst_events to disk")

burst_events_path = os.path.join(feeding_burst_output, f"burst_events_{VIDEO_RUN_NAME}.parquet")
burst_events.to_parquet(burst_events_path)
logging.info(f"saved burst events -> {burst_events_path}")


# ── STEP 5 — graph: burst rate, control vs feeding ────────────────────────────
banner("STEP 5 — graph: burst rate, control vs feeding")

burst_events = pd.read_parquet(burst_events_path)  # read back — decoupled from STEP 3's render loop
food_start_time = tracks.loc[tracks.frame_number == FOOD_START, "timestamp"].iloc[0]

burst_events["time_bin"] = (burst_events["timestamp"] // RATE_BIN_S) * RATE_BIN_S
bin_rate = burst_events.groupby("time_bin").size() / RATE_BIN_S  # bursts/sec, pooled across all fish

control_rate = bin_rate[bin_rate.index < food_start_time].mean()
feeding_rate = bin_rate[bin_rate.index >= food_start_time].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bin_rate.index, bin_rate.values, color="teal", marker="o", markersize=3)
ax.axvspan(bin_rate.index.min(), food_start_time, color="tab:blue", alpha=0.1, label="control")
ax.axvspan(food_start_time, bin_rate.index.max(), color="tab:orange", alpha=0.1, label="feeding")
split_frac = (food_start_time - bin_rate.index.min()) / (bin_rate.index.max() - bin_rate.index.min())
ax.axhline(control_rate, xmin=0, xmax=split_frac, color="tab:blue", linestyle="--",
           label=f"control\nmean {control_rate:.2f}/s")
ax.axhline(feeding_rate, xmin=split_frac, xmax=1, color="tab:orange", linestyle="--",
           label=f"feeding\nmean {feeding_rate:.2f}/s")
ax.set_xlabel("time (s)")
ax.set_ylabel("burst rate (bursts/s, pooled across fish)")
ax.set_title("Burst rate over time — control vs feeding")
ax.legend()
rate_path = os.path.join(feeding_burst_output, "burst_rate_over_time.png")
fig.savefig(rate_path, dpi=150)
logging.info(f"saved burst-rate evidence figure -> {rate_path}")
