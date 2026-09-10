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

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner
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

