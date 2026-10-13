"""
tracker.py - the fish tracker, Stage 4 core version.
Output: a per-run folder holding the annotated video, log, config sidecar and tracks.parquet

Run:  python -m scripts.tracker
"""

# imports

import yaml
import logging
import cv2
import mlflow
from ultralytics import YOLO

from scripts.console import banner, banner_sub
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS

CONFIG_PATH = 'config.yaml'
TRACKING_URI = 'sqlite:///mlflow.db' # the same store log_artifact_mlflow.py registers into

# MAIN

# ── STEP 1: load the config ────────────────────────────────────────────────────
banner("STEP 1 - LOAD config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
cfg = cfg["tracker"]
logger.info(cfg)

input_video_path  = cfg["input_video_path"]
model_path        = cfg["model_path"]
output_video_path = cfg["output_video_path"] # where the run folder goes
start_seconds     = cfg["start_seconds"]
end_seconds       = cfg["end_seconds"]
num_fish          = cfg["num_fish"] # how many IDs to lock (4)
calibration_secs  = cfg["calibration_secs"] # : how long to watch before locking on exisitng fish
max_distance      = cfg["max_distance"] # the furthest a fish can move between frames and still count as a match (150 px)

logger.info(f"\ninput_video_path: {input_video_path}")
logger.info(f"model_path: {model_path}")
logger.info(f"output_video_path: {output_video_path}")
logger.info(f"start_seconds: {start_seconds}")
logger.info(f"end_seconds: {end_seconds}")
logger.info(f"num_fish: {num_fish}")
logger.info(f"calibration_secs: {calibration_secs}")
logger.info(f"max_distance: {max_distance}")

# ── STEP 2: open the video ─────────────────────────────────────────────────────
banner("STEP 2 - OPEN the video")

# open the video with cv2
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"could not open video: {input_video_path}")

# set the playhead at start video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps)) # cv2 only understand frame number - here 0 x 60 = frame zero but if 5 seconds then 5 x 60 = frame 300 start
logger.info(f"fps: {fps}")

# read width and height
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # returns a float - later we need an integer
logger.info(f"frame size: {W} x {H}")

# ── STEP 3: load the YOLO model ────────────────────────────────────────────────
banner("STEP 3 - LOAD the YOLO model")

mlflow.set_tracking_uri(TRACKING_URI)
weights = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_path}/artifacts/best.pt") # models:/aquamind-yolo-detector@baseline/artifacts/best.pt # ask the registry for this alias's best.pt: alias → version → copy to a temp folder (never touch mlruns/ directly)
logger.info(f"weights downloaded to: {weights}")
model = YOLO(weights) # # build the YOLOv8 network + load the trained numbers from best.pt → a model we can run on frames (.boxes), unlike MLflow's pyfunc wrapper

# ── STEP 4: fish memory ────────────────────────────────────────────────────────
# one dict per fish: id, x, y, vx, vy, bbox, confidence, hits, missing

tracks  = []   # the tracker's memory: one dict per fish it is currently following
next_id = 1    # the ID number the next new fish will get


# ── STEP 5: the frame loop ─────────────────────────────────────────────────────

ok, frame = cap.read()
logger.info(f"ok {ok}") # ok True 0 - the playhead succesfully read the first set frame - a NumPy array of pixels
logger.info(f"frame shape: {frame.shape}")
cv2.imwrite("test_frame.jpg", frame) # wrote a file in the root ok 
results = model(frame, verbose=False) # object contain alll what yolo found in that frame...boxes, labels, ....etcetera
boxes = results[0].boxes  # all boxes in frame positrion 0
logger.info(f"boxes found: {len(boxes)}")


# while True: read a frame, stop at the end of the clip

    # STEP 5a: detect, run YOLO on the frame, keep class 0 (danio_rerio), skip class 1 (reflection)

    # STEP 5b: predict each fish's next position + match predictions to boxes (Hungarian)

    # STEP 5c: ghosts, healthy fish match first, missing fish search a wider (capped) radius

    # STEP 5d: calibration lock, after calibration_secs keep the num_fish best tracks, give IDs 1..N

    # STEP 5e: logging, occlusion_lost / occlusion_recovery + crossings (two fish's boxes overlapping)

    # STEP 5f: store this frame's rows for tracks.parquet + draw boxes on the frame


# ── STEP 6: save outputs ───────────────────────────────────────────────────────
# tracks.parquet + annotated video
