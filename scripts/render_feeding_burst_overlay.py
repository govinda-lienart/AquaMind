"""
render_feeding_burst_overlay.py — live per-fish burst overlay on the tracked video.

usage:  python -m scripts.render_feeding_burst_overlay
"""

# ── imports ──────────────────────────────────────────────────────────────────
import os


import pandas as pd

from scripts.video_utils import grab_video_name

# CONSTANTS

VIDEO_RUN_NAME = "IMG_2349_appearance_2026_08_12_1926"

# ── config ──────────────────────────────────────────────────────────────────




# ── STEP 1 — load tracks + resolve the video path ────────────────────────────
parquet_path, pixels_per_cm, *_ = grab_video_name(VIDEO_RUN_NAME)
tracks = pd.read_parquet(parquet_path)
run_dir = os.path.dirname(parquet_path)
video_path = os.path.join(run_dir, f"tracker_{VIDEO_RUN_NAME}.mp4")

# ── STEP 2 — per-fish speed + burst detection ────────────────────────────────
# same thresholds as feeding_activity.py: SMOOTH_WIN, BURST_MIN_CMS, BURST_PROM, BURST_GAP_S
# helper: bursts_up_to(fish_id, frame_number) -> running count
# helper: is_flickering(fish_id, frame_number) -> True if a burst just peaked





# ── STEP 3 — render the overlay video ────────────────────────────────────────
# open the tracked video, loop frames, draw counter + marker per fish, write output
