"""
Parses a LabelStudio YOLO export and stores bboxes in MySQL.

Input  : label .txt files from config labels_path
Needs  : frames already extracted and registered in MySQL
Output : one row in annotation_sets + rows in annotations table (bboxes)

Usage: python -m scripts.store_annotations
"""

# IMPORTS───────────────────────────────────────────

import os
import yaml
import datetime

# module imports
from scripts.console import banner, banner_sub
from scripts.db import get_connection, get_frame_id, get_video_id

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS───────────────────────────────────────────


# ── STEP 1: READ config.yaml + CONNECT ────────────────────────────────────────
banner("STEP 1 - READ config.yaml + CONNECT")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["store_annotations"]
logger.info(cfg)

banner_sub("load store_annotations section of config.yaml")
labels_path   = cfg["labels_path"]     # labelstudio_download/<project>_<timestamp>/labels
frames_folder = cfg["frames_folder"]   # frames/frames_IMG_0764_20260624_1635
video_name    = cfg["video_name"]      # "IMG_0350.MOV"
frame_source  = cfg["frame_source"]    # regular / crossing_event / ghosting_event (the sidecar can override it in Step 2)
notes         = cfg.get("notes", "")   # optional, so .get instead of []
logger.info(f"video_name={video_name}, frame_source={frame_source}, labels_path={labels_path}")

banner_sub("connect to MySQL")
conn = get_connection()
logger.info("connected to sql database")
reading_cursor = conn.cursor()   # SELECT queries (fetching)
insert_cursor  = conn.cursor()

# ── STEP 2: READ SIDECARS ─────────────────────────────────────────────────────
banner("STEP 2 - READ SIDECARS")



banner_sub("extraction_params.yaml (from the frames folder)")

banner_sub("download_params.yaml (next to labels/, from download_labelstudio.py)")

# ── STEP 3: CREATE annotation_sets ROW ────────────────────────────────────────
banner("STEP 3 - CREATE annotation_sets ROW")


# ── STEP 4: LOOP label files -> INSERT annotations ────────────────────────────s
banner("STEP 4 - LOOP label files -> INSERT annotations")


# ── STEP 5: COMMIT + DONE ─────────────────────────────────────────────────────
banner("STEP 5 - COMMIT + DONE")
