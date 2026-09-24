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

banner_sub("look up video_id (foreign key)")
video_id = get_video_id(reading_cursor, video_name)   # SELECT id FROM videos ... -> the FK for annotation_sets
logger.info(f"video_name={video_name} -> video_id={video_id}")

banner_sub("insert annotation_sets row")
insert_cursor.execute(
    """INSERT INTO annotation_sets
       (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
        sample_rate, start_seconds, end_seconds, created_at,
        ls_project_name, ls_project_id, ls_min_task_id, ls_max_task_id, ls_downloaded_at)
       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
    (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
     sample_rate, start_seconds, end_seconds, datetime.datetime.now(),
     ls_project_name, ls_project_id, ls_min_task_id, ls_max_task_id, ls_downloaded_at)
)
annotation_set_id = insert_cursor.lastrowid   # the id MySQL just generated for this row
logger.info(f"success creation of annotation_set_id={annotation_set_id}")

# ── STEP 4: LOOP label files -> INSERT annotations ────────────────────────────s
banner("STEP 4 - LOOP label files -> INSERT annotations")

banner_sub("loop over label files")
total_frames = 0
total_annotations = 0

for label_file in os.listdir(labels_path):        # one .txt per labelled frame
    if not label_file.endswith(".txt"):
        continue                                   # skip anything that isn't a label file

    # 4a. parse the frame number from the filename (the only clue the label file gives about which frame it belongs to)
    # filename: 8d4813a4-frame_0_IMG_0867.txt  ->  frame number 0 => cleaning the labelstudio format
    name = label_file.split("-")[1]                # "frame_0_IMG_0867.txt"  (drop the LabelStudio hash before the "-")
    stem = os.path.splitext(name)[0]               # "frame_0_IMG_0867"      (drop ".txt")
    frame_number = int(stem.split("_")[1])         # "0"                     (the part after "frame_") 

    logger.info(f"{label_file} -> frame_number={frame_number}") #  8d4813a4-frame_0_IMG_0867.txt -> frame_number=0

    # 4b. look up frame_id (FK), skip the file if the frame isn't in MySQL
    try:
        frame_id = get_frame_id(reading_cursor, frames_folder, frame_number)   # SELECT id FROM frames ... -> FK for annotations
    except ValueError:
        logger.warning(f"frame {frame_number} not in MySQL, skipping {label_file}")
        continue 
    total_frames += 1

    # 4c. read the label file (one line per bounding box)
    with open(os.path.join(labels_path, label_file)) as f:
        lines = f.readlines()                      # one line per bounding box

    for line in lines:
        # 4d. parse one bounding box
        tokens = line.split()[:5]                  # "0 0.44 0.66 0.05 0.06" -> ["0", "0.44", "0.66", "0.05", "0.06"]
        if len(tokens) != 5:
            logger.warning(f"{label_file}: expected 5 values, got {len(tokens)}, skipping line")
            continue

        class_id = int(tokens[0])
        x_center = float(tokens[1])
        y_center = float(tokens[2])
        width    = float(tokens[3])
        height   = float(tokens[4])

        # 4e. insert it, linked to its frame (frame_id) and its batch (annotation_set_id)
        insert_cursor.execute(
            "INSERT INTO annotations (frame_id, annotation_set_id, class_id, label, x_center, y_center, width, height, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (frame_id, annotation_set_id, class_id, LABEL_MAP[class_id],
             x_center, y_center, width, height, datetime.datetime.now())
        )
        total_annotations += 1

logger.info(f"frames processed={total_frames}, annotations inserted={total_annotations}")


# ── STEP 5: COMMIT + DONE ─────────────────────────────────────────────────────
banner("STEP 5 - COMMIT + DONE")

