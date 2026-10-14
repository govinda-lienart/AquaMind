"""
tracker.py - the fish tracker, Stage 4 core version.
Output: a per-run folder holding the annotated video, log, config sidecar and tracks.parquet

Run:  python -m scripts.tracker
"""

# imports

import os
import yaml
import logging
from datetime import datetime
import cv2
import math
import mlflow
from ultralytics import YOLO
import numpy as np


from scripts.console import banner, banner_sub
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS

CONFIG_PATH = 'config.yaml'
TRACKING_URI = 'sqlite:///mlflow.db' # the same store log_artifact_mlflow.py registers into
RUN_TAG      = 'L0'   # ablation level written into the run folder name (becomes automatic once the use_* flags exist)
COLOURS      = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # one BGR colour per fish ID (OpenCV = BGR, not RGB)

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

tracks  = []   # the tracker's memory: one dict per fish it is currently following ---after 5.2 birth after tracks = [ {"id": 1, "x": 716.1, "y": 561.1, "vx": 0, "vy": 0, "bbox": [656, 541, 776, 581], "missing": 0},....
next_id = 1    # the ID number the next new fish will get


# ── STEP 5: the frame loop ─────────────────────────────────────────────────────
banner("STEP 5 - FRAME LOOP: detect → match → update")

# 5.0 output: a fresh run folder per run + the annotated video writer
stamp   = datetime.now().strftime("%Y_%m_%d_%H%M")              # e.g. 2026_10_14_1530 → a new folder every run, never overwrite
run_dir = f"{output_video_path}_{RUN_TAG}_{stamp}"               # e.g. output_fish_tracker/tracker_IMG_1839_L0_2026_10_14_1530
os.makedirs(run_dir, exist_ok=True)                              # create the folder (no error if it already exists)
video_path = os.path.join(run_dir, "tracked.mp4")
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))  # size MUST match the frames exactly, or the file is silently broken
logger.info(f"run folder: {run_dir}")

# 5.0 counters for the end-of-run summary (the ablation numbers)
collisions   = 0    # a fish matched to a dot another fish already took this frame (two fish on one dot)
teleports    = 0    # a fish jumped further than max_distance in one frame (grabbed a wrong, far dot)
empty_frames = 0    # frames where YOLO found no fish at all
unmatched_detections = 0  # boxes YOLO found that no fish ID claimed (the grey-only boxes in the video)

# for every frame of the clip: find the fish with YOLO, link every fish in memory to its nearest new dot, update its position and velocity
max_frames = int((end_seconds - start_seconds) * fps) if end_seconds is not None else None  # frames in the clip (None = whole video), e.g. (20 - 0) x 59.8 = 1195
logger.info(f"max_frames: {max_frames}")
frame_number = 0                               # counter: which frame we're on (the loop reads frame 0 itself)
while True:
    if max_frames is not None and frame_number >= max_frames:  # reached end_seconds
        break
    ok, frame = cap.read()                     # grab the frame under the playhead, playhead moves on (ok = False at end of video)
    if not ok:                                 # video finished before end_seconds
        break                                  # jump out of the while loop

    # 5.1 detect # where are the fish in this picture?"
    results = model(frame, verbose=False)      # object containing all YOLO found in this frame: boxes, labels, ... results[0] = the only image we gave it
    boxes = results[0].boxes                   #0 means the first  image I gave YOLO  # results=  all boxes YOLO found (PyTorch tensors: .cls, .conf, .xywh → .tolist() to read them)
    fish_boxes = boxes[boxes.cls == 0]         # keep fish (class 0), drop reflections (class 1) - boolean mask, like df[mask]
    new_centres = fish_boxes.xywh[:, :2].tolist()  # x y - [x_centre, y_centre, width, height] per fish ->[:, :2] = [rows, columns] [:, :2] keeps x, y
    new_bboxes  = fish_boxes.xyxy.tolist()     # for drwawing the boxes - [x1, y1, x2, y2] corners per fish, SAME order as new_centres (entry i = same box)
    # SORT matches by box overlap, which works for large, slow objects like pedestrians. My fish are small, thin and fast relative to their body size,
    # so boxes often stop overlapping between frames and IoU collapses to zero. Centre distance to a motion-predicted position degrades gracefully, so I used that instead.
    if not new_centres:
        empty_frames += 1 # it counts the frames where YOLO found zero fish.



"""-------------------------------------"""

    if not tracks: # if tracks is still empty, create the fish
        # 5.2  birth - First frame: "give each fish a name" - first frame: memory is empty → one new fish per centre (initialisation, runs once)
        if len(new_centres) != num_fish:       # fail fast: a bad start frame would create phantom / missing fish
            raise ValueError(f"start frame has {len(new_centres)} fish boxes, expected {num_fish}: pick another start_seconds")
        for (x, y), bbox in zip(new_centres, new_bboxes):   # each centre + its box found in the first frame (zip = walk both lists side by side)
            tracks.append({"id": next_id, "x": x, "y": y, "vx": 0, "vy": 0, "bbox": bbox, "missing": 0})  # a new fish is born
            next_id += 1                       # next fish gets the next number
        for fish in tracks:
            logger.info(f"{fish}")             # one fish per line
    else:
        # 5.3 match + update (L0: each fish takes its nearest new dot) "which box is which fish now?"
        taken = set()                          # dots already claimed this frame (to count collisions)
        for fish in tracks:                    # each fish in memory
            best_d = float("inf")              # clean whiteboard: smallest distance so far
            best_c = None                      # ...and the dot that gave it
            best_b = None                      # ...and that dot's box
            for (nx, ny), bbox in zip(new_centres, new_bboxes):  # each new dot + its box
                d = math.hypot(nx - fish["x"], ny - fish["y"])  # distance from where the fish was (Pythagoras)
                if d < best_d:                 # closer than the best so far?
                    best_d = d                 # remember the new smallest distance
                    best_c = (nx, ny)          # remember which dot
                    best_b = bbox              # remember its box
            if best_c is None:                 # no fish boxes this frame → nothing to match, keep the old position
                continue                       # skip THIS fish (not the whole frame)
            if best_c in taken:                # another fish already took this dot this frame
                collisions += 1
            taken.add(best_c)
            if best_d > max_distance:          # jumped further than a fish can swim in one frame
                teleports += 1
            fish["vx"] = best_c[0] - fish["x"] # sideways move this frame (+ = right)
            fish["vy"] = best_c[1] - fish["y"] # vertical move this frame (+ = down)
            fish["x"], fish["y"] = best_c      # new position becomes the memory
            fish["bbox"] = best_b              # ...and so does its box (for drawing)
        unmatched_detections += len(new_centres) - len(taken)   # boxes nobody took this frame

    # 5.4 draw + write this frame to the video "show what the tracker decided"  grey boxes: what YOLO saw - loured boxes + F1–F5 labels: what the tracker decided
    for (x1, y1, x2, y2) in fish_boxes.xyxy.tolist():                 # YOLO's fish boxes: thin grey (what the detector saw)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 1)
    for fish in tracks:                                               # the tracker's fish: coloured box + dot + ID (what the tracker decided)
        colour = COLOURS[(fish["id"] - 1) % len(COLOURS)]             # same ID → same colour every frame, so a swap is easy to see
        x1, y1, x2, y2 = (int(v) for v in fish["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)           # thick coloured box = the box this fish claimed
        cx, cy = int(fish["x"]), int(fish["y"])
        cv2.circle(frame, (cx, cy), 6, colour, -1)                    # -1 = filled circle
        cv2.putText(frame, f"F{fish['id']}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)  # label above the box's top-left corner
    cv2.putText(frame, f"frame {frame_number}  {RUN_TAG}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # frame counter: find crossings by frame later
    out.write(frame)

    if frame_number % int(fps) == 0:           # log once per second of video, not every frame
        logger.info(f"frame {frame_number}: " + ", ".join(f"F{f['id']} ({f['x']:.0f},{f['y']:.0f})" for f in tracks))  # one line: every fish's position this frame
    frame_number += 1                          # move the counter on

cap.release()                                  # close the input video
out.release()                                  # close the output video (without this the .mp4 can be unreadable)
logger.info(f"processed {frame_number} frames")
logger.info(f"video saved: {video_path}")

banner(f"SUMMARY - {RUN_TAG}")
logger.info(f"frames processed : {frame_number}")
logger.info(f"empty frames     : {empty_frames}   (YOLO found no fish)")
logger.info(f"collisions       : {collisions}   (two fish matched to one dot)")
logger.info(f"teleports        : {teleports}   (jump > max_distance = {max_distance} px)")
logger.info(f"unmatched dets   : {unmatched_detections}   (YOLO saw a fish, no ID claimed it)")


# ── TODO: core tracker still to build (commented out so they don't print yet) ──
# banner_sub("5.7 - prediction: match against x + vx, not last position")
# banner_sub("5.8 - velocity smoothing: EMA so one jittery box can't throw the prediction")
# banner_sub("5.9 - one dot, one fish: Hungarian matching (linear_sum_assignment)")
# banner_sub("5.10 - distance gate: refuse matches beyond max_distance (no teleporting)")
# banner_sub("5.11 - missing fish (ghosts): missing += 1, coast on prediction, recover")
# banner_sub("5.12 - calibration window: wait for all num_fish before locking IDs")
# banner_sub("5.13 - outputs: tracks.parquet, annotated video, crops, run folder")
# banner_sub("5.14 - refactor: dicts → Track class, matching → associate()")
