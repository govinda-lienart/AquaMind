"""
tracker.py - the fish tracker, Stage 4 core version.
Output: a per-run folder holding the annotated video, log, config sidecar and tracks.parquet

Run:  python -m scripts.tracker
"""

# imports

import yaml
import logging

from scripts.console import banner, banner_sub
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS

CONFIG_PATH = 'config.yaml'

# MAIN

# ── STEP 1: load the config ────────────────────────────────────────────────────
banner("STEP 1 - LOAD config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
cfg = cfg["tracker"]
logger.info(cfg)

# 1b. pull out the settings
input_video_path  = cfg["input_video_path"]
model_path        = cfg["model_path"]
output_video_path = cfg["output_video_path"] # where the run folder goes
start_seconds     = cfg["start_seconds"]
end_seconds       = cfg["end_seconds"]
num_fish          = cfg["num_fish"] # how many IDs to lock (4)
calibration_secs  = cfg["calibration_secs"] # : how long to watch before locking on exisitng fish
max_distance      = cfg["max_distance"] # the furthest a fish can move between frames and still count as a match (150 px)




# ── STEP 2: open the video ─────────────────────────────────────────────────────
# open the video with cv2, read fps / width / height


# ── STEP 3: load the YOLO model ────────────────────────────────────────────────
# load_yolo() from scripts.model_registry understands models:/aquamind-yolo-detector@baseline


# ── STEP 4: fish memory ────────────────────────────────────────────────────────
# one dict per fish: id, x, y, vx, vy, bbox, confidence, hits, missing


# ── STEP 5: the frame loop ─────────────────────────────────────────────────────
# while True: read a frame, stop at the end of the clip

    # STEP 5a: detect, run YOLO on the frame, keep class 0 (danio_rerio), skip class 1 (reflection)

    # STEP 5b: predict each fish's next position + match predictions to boxes (Hungarian)

    # STEP 5c: ghosts, healthy fish match first, missing fish search a wider (capped) radius

    # STEP 5d: calibration lock, after calibration_secs keep the num_fish best tracks, give IDs 1..N

    # STEP 5e: logging, occlusion_lost / occlusion_recovery + crossings (two fish's boxes overlapping)

    # STEP 5f: store this frame's rows for tracks.parquet + draw boxes on the frame


# ── STEP 6: save outputs ───────────────────────────────────────────────────────
# tracks.parquet + annotated video
