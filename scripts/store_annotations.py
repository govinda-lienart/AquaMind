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

LABEL_MAP = {0: "danio_rerio", 1: "reflection"}

# ── STEP 1: READ config.yaml + CONNECT ────────────────────────────────────────
banner("STEP 1 - READ config.yaml + CONNECT")

banner_sub("load store_annotations section of config.yaml")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["store_annotations"]
logger.info(cfg)

labels_path   = cfg["labels_path"]     # labelstudio_download/<project>_<timestamp>/labels
frames_folder = cfg["frames_folder"]   # frames/frames_IMG_0764_20260624_1635
video_name    = cfg["video_name"]      # "IMG_0350.MOV"
frame_source  = cfg["frame_source"]    # regular / crossing_event / ghosting_event (the sidecar can override it in Step 2)
notes         = cfg.get("notes", "")   # optional, so .get instead of []
logger.info(f"video_name={video_name}, frame_source={frame_source}, labels_path={labels_path}")

#__

banner_sub("connect to MySQL")
conn = get_connection()
logger.info(f"connected to sql database: {conn.is_connected()}")
reading_cursor = conn.cursor()   # SELECT queries (fetching)
insert_cursor  = conn.cursor()   # INSERTs

# ── STEP 2: READ 2 SIDECARS (EXTRACT + DOWNLOAD LS) ─────────────────────────────────────────────────────
banner("STEP 2 - READ SIDECARS")

banner_sub("extraction_params.yaml (from the frames folder) => produced by extract.frames.py")  
extraction_path = os.path.join(frames_folder, "extraction_params.yaml") # .get() needs a dictionary first, and it only handles a missing KEY, not a missing FILE
if os.path.exists(extraction_path):
    with open(extraction_path) as f:
        sidecar = yaml.safe_load(f) or {}   # "or {}" covers a file that exists but is empty (safe_load returns None) => safe_load retuns None but its not a dictionary...so later the script might crash when we use .get so better to return {}
else:
    sidecar = {}                            # file missing -> empty dict, so every .get() below gives None
    logger.warning(f"no sidecar found at {extraction_path}")
logger.info(f"sidecar: {sidecar}")

frame_source     = sidecar.get("frame_source", frame_source)   # we use get here - returns None if abscent and the script carries on. That None goes into MySQL as NULL. a sidecar["frame_source"] that is empty leads to error - and very likely several parameters empty depending which extrraction script we have used
frames_extracted = sidecar.get("frames_extracted")
iou_threshold    = sidecar.get("iou_threshold")     # crossing frames only
dedup_window     = sidecar.get("dedup_window")      # crossing frames only
sample_rate      = sidecar.get("sample_rate")
start_seconds    = sidecar.get("start_seconds")
end_seconds      = sidecar.get("end_seconds")

#__

banner_sub("download_params.yaml (next to labels/, from download_labelstudio.py)")
download_path = os.path.join(os.path.dirname(labels_path), "download_params.yaml")   # parent of labels/ = the download folder
if os.path.exists(download_path):
    with open(download_path) as f:
        dl_sidecar = yaml.safe_load(f) or {}   # "or {}" covers an empty file (safe_load returns None)
else:
    dl_sidecar = {}                            # file missing (e.g. an older download) -> empty dict, every .get() below gives None
    logger.warning(f"no sidecar found at {download_path}")
logger.info(f"dl_sidecar: {dl_sidecar}")

ls_project_name  = dl_sidecar.get("project_name")     # name of the project in LabelStudio
ls_project_id    = dl_sidecar.get("project_id")       # project id in LabelStudio
ls_min_task_id   = dl_sidecar.get("min_task_id")      # first labeled task id exported
ls_max_task_id   = dl_sidecar.get("max_task_id")      # last labeled task id exported
ls_downloaded_at = dl_sidecar.get("downloaded_at")    # when the labels were downloaded from LabelStudio

# ── STEP 3: CREATE annotation_sets ROW ────────────────────────────────────────
banner("STEP 3 - CREATE annotation_sets ROW")


# ── STEP 4: LOOP label files -> INSERT annotations ────────────────────────────s
banner("STEP 4 - LOOP label files -> INSERT annotations")


# ── STEP 5: COMMIT + DONE ─────────────────────────────────────────────────────
banner("STEP 5 - COMMIT + DONE")
