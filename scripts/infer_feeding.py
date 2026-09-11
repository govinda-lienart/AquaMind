"""
infer_feeding.py — run the trained feeding-strike model over video as a sliding-window detector.

Output: <run>/feeding_train_test/output_infer/feeding_predictions_<stamp>.parquet
        columns: fish_id, segment, frame_start, frame_end, score, pred

usage:  python -m scripts.infer_feeding
"""
import os
import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from scripts.reid_features import transform, load_backbone # load_backbone(loads torch.hub.load("facebookresearch/dinov2", name))
from PIL import Image


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
banner("── STEP 4 — build the list of windows to score")
run_dir   = os.path.dirname(parquet_path)          # the tracker run folder
crops_dir = os.path.join(run_dir, "crops")
fish_ids = sorted(tracks["fish_id"].unique())
logger.info(f"fish ids: {fish_ids}")

# frames each fish actually has a CROP FILE for — the gap check must trust the .jpgs on disk,
# not tracks: the tracker logs positions for more frames than it wrote crops for (occlusion/merge
# frames get a track row but no crop), so checking tracks lets through windows whose crops are missing.
frames_by_fish = {}
for fid in fish_ids:
    files = glob.glob(os.path.join(crops_dir, f"fish_{fid}", f"frame_*_fish_{fid}.jpg"))
    frames_by_fish[fid] = {
        int(re.search(r"frame_(\d+)_", os.path.basename(f)).group(1)) for f in files
    }
    logger.info(f"fish {fid}: {len(frames_by_fish[fid])} crop files on disk")

# frames_by_fish ends up like:
#   {1: {7505, 7506, 7507, 7509, ...},   # 7508 missing = tracker gap for fish 1
#    2: {7505, 7506, 7507, 7508, ...},
#    3: {...}, 4: {...}}


windows = [] # list of tuples
"""windows = [
    ("before_food", 1, 7505, 7549),
    ("before_food", 1, 7525, 7569),
    ..."""
    
for seg_name, (seg_start, seg_end) in SEGMENTS.items(): # for each of the 2 segments
    for fish_id in fish_ids: # for each fish#
        have = frames_by_fish.get(fish_id, set())   # this fish's tracked frame numbers, looked up once
        for start in range(seg_start, seg_end - WINDOW + 1, STRIDE): # for each sliding start position → make one window
            # start = makes a list of starting points, spaced STRIDE apart # range(7505, 21477 - 45 + 1, 20) produces  7505, 7525, 7545, 7565, 7585, 7605, ... up to ~21430
            end = start + WINDOW - 1 # minus 1 because the start frame counts as frame 1 of the 45.
            if not all(f in have for f in range(start, end + 1)):  # all(...) returns True only if every item is true. 
                continue   # tracker lost this fish somewhere in the window — skip, crops would be missing
            windows.append((seg_name, fish_id, start, end))

logger.info(f"built {len(windows)} candidate windows (windows with tracker gaps dropped)")
for seg_name in SEGMENTS:
    n = sum(1 for w in windows if w[0] == seg_name) # stream of 1, 1, 1, ... for eacgh segnebt
    logger.info(f"  {seg_name}: {n}")

banner_sub("first 3 before_food windows")
sand_windows = [x for x in windows if x[0] == "before_food"]
for w in sand_windows[:3]:
    logger.info(f"{w}")    

# ── STEP 5 — embed every window's crops with DINOv2 ────────────────────────
# load the 45 crop images per window, transform, batch through the backbone (EMB_BATCH)
# result: one (45, 384) tensor per window

# helper function
def crop_paths_for(fish_id, start, end):
    """Build the ordered list of crop-image paths for one window (one .jpg per frame, start..end inclusive)."""
    fish_dir = os.path.join(crops_dir, f"fish_{fish_id}")
    return [os.path.join(fish_dir, f"frame_{f}_fish_{fish_id}.jpg") for f in range(start, end + 1)]
    # e.g. [".../crops/fish_1/frame_7505_fish_1.jpg", ".../crops/fish_1/frame_7506_fish_1.jpg", ...]

banner("── STEP 5 — embed every window's crops with DINOv2")
logger.info(f"{len(windows)} windows -> {len(windows) * WINDOW} crop forwards through DINOv2")

t_start = datetime.now()
window_embs = []                                       # one (45, 384) tensor per window, same order as `windows`
for i, (seg, fid, start, end) in enumerate(windows, start=1):     # i = running counter 1..N
    paths = crop_paths_for(fid, start, end)             # 45 crop-file paths for this window
    imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])   # (45, 3, 224, 224)
    with torch.no_grad():
        emb = backbone(imgs.to(DEVICE))     # (45, 384)
    window_embs.append(emb.cpu())           # list of (45, 384) tensors
    if i % 25 == 0 or i == len(windows):
        elapsed = (datetime.now() - t_start).total_seconds()
        rate = i / elapsed                                    # windows / sec
        eta = (len(windows) - i) / rate                       # seconds left
        logger.info(f"  window {i:4d}/{len(windows)}  ({100*i/len(windows):4.1f}%)  "
                    f"seg={seg:11s} fish={fid}  {rate:4.1f} win/s  eta {eta:5.0f}s")

logger.info(f"embedded {len(window_embs)} windows in {(datetime.now() - t_start).total_seconds():.0f}s")
logger.info(f"first window emb shape: {tuple(window_embs[0].shape)}")   # expect (45, 384)

# ── STEP 6 — run the LSTM over every window ────────────────────────────────
# forward pass, softmax, take P(strike); pred = score >= PROB_THRESHOLD
banner("── STEP 6 — score every window (LSTM + head → P(strike))")
results = []
for i, ((seg, fid, start, end), emb) in enumerate(zip(windows, window_embs), start=1):  # i = running counter 1..N
    x = emb.unsqueeze(0).to(DEVICE)                        # (45,384) -> (1,45,384): add the batch axis
    with torch.no_grad():
        logits = model(x)                                 # (1, 2) — raw scores [no_strike, strike]
    prob = torch.softmax(logits, dim=1)[0, 1].item()      # P(strike), a plain float 0..1
    pred = int(prob >= PROB_THRESHOLD)
    results.append((seg, fid, start, end, prob, pred))
    if i % 200 == 0 or i == len(windows):
        logger.info(f"  scored {i:4d}/{len(windows)} windows")


# ── STEP 7 — assemble and write the predictions parquet ────────────────────
# one row per window: fish_id, segment, frame_start, frame_end, score, pred
# write to <run>/feeding_train_test/output_infer/feeding_predictions_<RUN_STAMP>.parquet

banner("── STEP 7 — assemble and write the predictions parquet")
pred_df = pd.DataFrame(results, columns=["segment", "fish_id", "frame_start", "frame_end", "score", "pred"])
logger.info(pred_df.head().to_string())
out_dir = os.path.join(run_dir, "feeding_train_test", "output_infer")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"feeding_predictions_{RUN_STAMP}.parquet")
pred_df.to_parquet(out_path)
logger.info(f"wrote {len(pred_df)} rows → {out_path}")

# ── STEP 8 — quick summary to the log ─────────────────────────────────────
# positive-window count per segment, side by side — the negative-control readou 
banner("── STEP 8 — negative-control readout")

# per segment: how many windows the model flagged as strike, and the mean P(strike)
for seg_name in SEGMENTS:
    sub = pred_df[pred_df["segment"] == seg_name]
    n_pos = int(sub["pred"].sum())
    rate = 100 * n_pos / len(sub)                     # % of windows flagged positive — segment totals differ (crop gaps), so a raw count isn't a fair comparison
    logger.info(f"{seg_name:12s}  positives {n_pos:4d} / {len(sub):4d}  ({rate:5.1f}%)   mean P(strike) {sub['score'].mean():.3f}")

# per fish breakdown (matches the numbers quoted in feeding_strike_findings.md)
banner_sub("positive windows per segment x fish")
logger.info(pred_df.groupby(["segment", "fish_id"])["pred"].sum().to_string())