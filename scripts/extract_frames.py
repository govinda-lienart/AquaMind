"""
Extracts/stores frames (typically - one frame per second) from a video and stores the frame paths in MySQL.

Usage: python -m scripts.extract_frames
"""

# ── STEP 0: IMPORTS ───────────────────────────────────────────────────────────
# (we add each import only when the step that needs it appears)

import os
import yaml
import datetime

# module imports
from scripts.console import banner, banner_sub
from scripts.db import get_connection

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── STEP 1: READ config.yaml ──────────────────────────────────────────────────
# DONE: load config.yaml, keep ['extract_frames'], pull out the five values, log them

banner("STEP 1 - READ config.yaml")

banner_sub("load extract_frames section of config.yaml")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["extract_frames"]
logger.info(cfg)

banner_sub("extract_frames specific parameters")
video_path = cfg["video_path"]
frames_dir = cfg["frames_dir"]
sample_rate = cfg["sample_rate"]
start_seconds = cfg["start_seconds"]
end_seconds = cfg["end_seconds"]

logger.info(f"video_path={video_path}, frames_dir={frames_dir}, sample_rate={sample_rate}, start_seconds={start_seconds}, end_seconds={end_seconds}")

# ── STEP 2: BUILD the output folder path ──────────────────────────────────────

banner("STEP 2 - BUILD output folder path")

video_filename = os.path.basename(video_path)        # 'videos/IMG_1839.MOV' -> 'IMG_1839.MOV'
video_name = os.path.splitext(video_filename)[0]     # 'IMG_1839.MOV' -> ('IMG_1839', '.MOV') -> 'IMG_1839'
logger.info(f"video_name={video_name}")
format_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
logger.info(f"format_now={format_now}")
frame_folder_path = f"{frames_dir}/frames_{video_name}_{format_now}"   # -> 'frames/frames_IMG_1839_20260919_1035'
logger.info(f"frame_folder_path={frame_folder_path}")

# ── STEP 3: CONNECT to MySQL ──────────────────────────────────────────────────

banner("STEP 3 - CONNECT to MySQL")

conn = get_connection()
logger.inf(f"connected to database={conn.database}")
cursor = conn.cursor()

banner_sub("guard 1 - unique constraint on frames (video_id, frame_number)") # never allow the same frame of the same video to be stored twice
try:
    cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
    logger.info("contrain added")
except Exception:
    logger.info("constrain already exist, continuing") # eats the error in case constrains already established





# ── STEP 4: GUARDRAILS (run in order) ─────────────────────────────────────────
# TODO: banner("STEP 4 - GUARDRAILS"), then banner_sub("guard 1 - unique constraint") / ("guard 2 - video registered") / ("guard 3 - already extracted?")
# TODO guard 1: unique constraint on frames (video_id, frame_number)
# TODO guard 2: get_video_id(cursor, video_path)
# TODO guard 3: frames already extracted? -> stop if yes


# ── STEP 5: EXTRACT frames to disk ────────────────────────────────────────────
# TODO: banner("STEP 5 - EXTRACT frames to disk"), banner_sub("video settings: fps / step / start-end frame"), banner_sub("saving frames")
# TODO: open video with cv2, work out fps / step / start_frame / end_frame
# TODO: loop, save every Nth frame as JPG into frame_folder_path
# TODO: count frames_stored


# ── STEP 6: REGISTER frames in MySQL ──────────────────────────────────────────
# TODO: banner("STEP 6 - REGISTER frames in MySQL")
# TODO: scan frame_folder_path, INSERT IGNORE one row per JPG into frames


# ── STEP 7: SIDECAR + DONE ────────────────────────────────────────────────────
# TODO: banner("STEP 7 - SIDECAR + DONE")
# TODO: write extraction_params.yaml into frame_folder_path
# TODO: log the summary
