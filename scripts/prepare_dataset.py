"""
Queries MySQL for annotated frames and builds a YOLO-ready dataset folder.

Input  : annotation_set_ids + dataset_name from config.yaml
Needs  : annotations stored in MySQL res
Output : dataset/<name>/images + labels (train/val), dataset_card.yaml, dataset.yaml (project root)

Usage: python -m scripts.prepare_dataset
"""

# IMPORTS───────────────────────────────────────────

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

# CONSTANTS───────────────────────────────────────────

# TODO: TRAIN_SPLIT, RANDOM_SEED, CLASS_NAMES

# ── STEP 1: READ config.yaml + CONNECT ────────────────────────────────────────
# banner("STEP 1 - READ config.yaml + CONNECT")
# 1a. load the prepare_dataset section of config.yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["prepare_dataset"]
logger.info(cfg)

dataset_name   = cfg["dataset_name"]     
annotation_set_ids = cfg["annotation_set_ids"]  

# 1b. pull out annotation_set_ids and dataset_name
# 1c. connect to MySQL + open a reading cursor

# ── STEP 2: FETCH ANNOTATED FRAMES ────────────────────────────────────────────
# banner("STEP 2 - FETCH ANNOTATED FRAMES")

# 2a. build the placeholders string: one %s per annotation_set_id
# 2b. SELECT DISTINCT frame id + frame_path (annotations JOIN frames, filtered by annotation_set_id IN (...))
# 2c. fetchall -> list of (frame_id, frame_path), log how many

# ── STEP 3: SPLIT TRAIN / VAL ─────────────────────────────────────────────────
# banner("STEP 3 - SPLIT TRAIN / VAL")

# 3a. seed the random generator (why? same split every run)
# 3b. shuffle the frames list
# 3c. compute the split index, slice into train_frames and val_frames

# ── STEP 4: CREATE YOLO FOLDERS ───────────────────────────────────────────────
# banner("STEP 4 - CREATE YOLO FOLDERS")

# 4a. dataset_path = dataset/<dataset_name>
# 4b. if it already exists -> delete it (fresh rebuild)
# 4c. create images/train, images/val, labels/train, labels/val

# ── STEP 5: COPY IMAGES + WRITE LABEL FILES ───────────────────────────────────
# banner("STEP 5 - COPY IMAGES + WRITE LABEL FILES")

# 5a. loop over the two splits: ("train", train_frames), ("val", val_frames)
# 5b.   loop over frames in that split
# 5c.     copy the image into images/<split>/
# 5d.     SELECT this frame's bboxes (class_id, x_center, y_center, width, height), same annotation_set_id filter
# 5e.     write labels/<split>/<frame_name>.txt, one "class x y w h" line per bbox
# 5f. log images copied + label lines written

# ── STEP 6: COLLECT METADATA FOR THE DATASET CARD ─────────────────────────────
# banner("STEP 6 - COLLECT METADATA")

# 6a. open a dictionary cursor (rows come back as dicts -> readable in the yaml)
# 6b. SELECT the videos that contributed frames (videos JOIN frames JOIN annotations)
# 6c. SELECT the annotation_sets rows used (provenance: frame_source, sample_rate, LS project...)
# 6d. get the current git commit hash (which code version built this dataset)

# ── STEP 7: WRITE dataset_card.yaml + dataset.yaml ────────────────────────────
# banner("STEP 7 - WRITE dataset_card.yaml + dataset.yaml")

# 7a. build the card dict (name, ids, git commit, videos, annotation sets, counts, classes, seed, created_at)
# 7b. yaml.dump it to dataset/<name>/dataset_card.yaml
# 7c. build the YOLO dict (path, train, val, nc, names)
# 7d. yaml.dump it to dataset.yaml at project root (what YOLO training reads)

# ── STEP 8: SUMMARY + CLOSE ───────────────────────────────────────────────────
# banner("STEP 8 - SUMMARY + CLOSE")

# 8a. log dataset name, ids, git commit, total / train / val, output path
# 8b. close cursors + connection
