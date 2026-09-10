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
from scripts.reid_features import transform, load_backbone # load_backbone(loads torch.hub.load("facebookresearch/dinov2", name))


import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name

import torch.nn as nn

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

# building the network using the class
class FeedingLSTMClassifier(nn.Module):
    def __init__(self, input_size =384, hidden_size=64, num_classes=2): # constructor - build and store the networks layers # input_size is tensor size of one frame after DINO2 processing # 64 is hidden state running summary afterbeing processed by LSTM.....
        """ what the network is made of"""
        super().__init__()  # = nn.Module.__init__(self) — run the parent's constructor  # creates an empty organized box
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # creates LSTM Layer and stores it in the organized box for easy access
        self.head = nn.Linear(hidden_size, num_classes) # received the function value arguments # expect a 64-long vector in, produce a 2-long vector out."
    def forward(self, x): # calls forward wehn call model(x) ---->> x model calls Feeding
        """what the network does with an input - blueprint 45 frame in - LSTM 64 summary - 2 scores"""
        output, (h_n, c_n) = self.lstm(x) # run the window through lstm 
        last_hidden = h_n[-1] # take its final memory vecotr (64 d vector)
        logits = self.head(last_hidden) # run the vinal memory (throught the head) 
        return logits            # 2 scores                       

# ── STEP 3 — load the trained checkpoint + the frozen DINOv2 backbone ──────
# model.load_state_dict(...), model.eval(); backbone in eval mode, no grad
banner("── STEP 3 — load checkpoint + DINOv2 backbone")
model = FeedingLSTMClassifier().to(DEVICE) # # box 1: lstm + head    (trained, your weights) runs inti (builds LSTM with random weights) - claculation on mps # crating box
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)) # put all the values/parameteres/weights nicely in the box we created in the class
model.eval() #switching to inference mode 
backbone = load_backbone(BACKBONE_NAME, device=DEVICE) # box 2: DINOv2          (frozen, pretrained, never touched)

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
