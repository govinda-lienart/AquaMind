"""
infer_feeding.py — run the trained feeding-strike model over video as a sliding-window detector.

Output: <run>/feeding_train_test/output_infer/feeding_predictions_<stamp>.parquet
        columns: fish_id, segment, frame_start, frame_end, score, pred

usage:  python -m scripts.infer_feeding
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name


# ── config ──────────────────────────────────────────────────────────────────
VIDEO_RUN_NAME  = "IMG_2349_appearance_2026_08_12_1926"
CHECKPOINT_PATH = "best_feeding_lstm.pt"
BACKBONE_NAME   = "dinov2_vits14"
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"

WINDOW, STRIDE  = 45, 20
PROB_THRESHOLD  = 0.5
EMB_BATCH       = 128

SAND_START, FOOD_START = 7505, 14491 # sand phase is the negative: water was injected but no food, so correct strike count is 0.
PHASE_LEN = FOOD_START - SAND_START # food phase clipped to the same length as the sand phase to keep the two counts comparable.

SEGMENTS = {
    "before_food": (SAND_START, FOOD_START),                 # sand-injection control, no food
    "after_food":  (FOOD_START, FOOD_START + PHASE_LEN),      # real feeding
}

RUN_STAMP = datetime.now().strftime("%Y_%m_%d_%H%M")


# STEP 1 — resolve paths and load the tracker output
# find the run folder, read tracks.parquet, work out the video name / fps
banner("── STEP 1 — resolve paths and load the tracker output")
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME)
tracks = pd.read_parquet(parquet_path)
banner_sub("laoding tracks")
logger.info(f"loading {len(tracks)} from {parquet_path}")
logger.info(tracks.head())

# ── STEP 2 — define the model blueprint ────────────────────────────────────
# the fixed-window LSTM class, matching train_feeding_lstm_fixed_window.py


# ── STEP 3 — load the trained checkpoint + the frozen DINOv2 backbone ──────
# model.load_state_dict(...), model.eval(); backbone in eval mode, no grad


# ── STEP 4 — build the list of windows to score ────────────────────────────
# for each segment, for each fish_id: slide (WINDOW, STRIDE) over the frame range
# keep only windows where the fish has a full 45 frames of track


# ── STEP 5 — embed every window's crops with DINOv2 ────────────────────────
# load the 45 crop images per window, transform, batch through the backbone (EMB_BATCH)
# result: one (45, 384) tensor per window


# ── STEP 6 — run the LSTM over every window ────────────────────────────────
# forward pass, softmax, take P(strike); pred = score >= PROB_THRESHOLD


# ── STEP 7 — assemble and write the predictions parquet ────────────────────
# one row per window: fish_id, segment, frame_start, frame_end, score, pred
# write to <run>/feeding_train_test/output_infer/feeding_predictions_<RUN_STAMP>.parquet


# ── STEP 8 — quick summary to the log ─────────────────────────────────────
# positive-window count per segment, side by side — the negative-control readout
