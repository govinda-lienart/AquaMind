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
import random
import shutil

# module imports
from scripts.console import banner, banner_sub
from scripts.db import get_connection

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS──────────────────────────────────────────
TRAIN_SPLIT = 0.8  # 80% of frames go to train, the other 20% to val
RANDOM_SEED = 42  # fixed seed, so the shuffle gives the same split every run

# ── STEP 1: READ config.yaml + CONNECT ────────────────────────────────────────
banner("STEP 1 - READ config.yaml + CONNECT")

# 1a. load the prepare_dataset section of config.yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["prepare_dataset"]
logger.info(cfg)

# 1b. pull out annotation_set_ids and dataset_name
timestamp    = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")   # e.g 2026_10_02_14h05
dataset_name = f"{cfg['dataset_name']}_{timestamp}"                 # e.g regular_data_ann_5r_9r_10r_2026_10_02_14h05
annotation_set_ids = cfg["annotation_set_ids"]  

# 1c. connect to MySQL + open a reading cursor
banner_sub("connect to MySQL")
conn = get_connection()
logger.info(f"connected to sql database: {conn.is_connected()}")
reading_cursor = conn.cursor()   # SELECT queries (fetching)

# ── STEP 2: FETCH ANNOTATED FRAMES ────────────────────────────────────────────
banner("STEP 2 - FETCH ANNOTATED FRAMES")

# 2a. fetch every labelled frame in the chosen annotation sets 
placeholders = ",".join(["%s"] * len(annotation_set_ids))
reading_cursor.execute(
    f"SELECT DISTINCT frames.id, frames.frame_path FROM frames JOIN annotations ON frames.id = annotations.frame_id WHERE annotations.annotation_set_id IN ({placeholders})", annotation_set_ids) # e.g IN (5, 9, 10)
frames = reading_cursor.fetchall() # fetching the selection - list of tuples   [                                      # ← the list of tuples => e.g frames = [(336, "frames/.../frame_1800_IMG_0909.jpg"), ...]
logger.info(f"total number of selected frames and its paths: {len(frames)}")

# ── STEP 3: SPLIT TRAIN / VAL ─────────────────────────────────────────────────
banner("STEP 3 - SPLIT TRAIN / VAL")
# 3a. seed the random generator + shuffle the frames list
random.seed(RANDOM_SEED)
random.shuffle(frames) # nshuffle changes the list in place and returns nothing (None) - is impaacted by the seed value - reproducitbility

# 3b compute the split index, slice into train_frames and val_frames
split_at = int(len(frames) * TRAIN_SPLIT) # e.g 260 * 0.8 = 208
train_frames = frames[:split_at] # e.g  # first 208 frames (positions 0-207)
val_frames = frames[split_at:] # # remaining 52 frames (position 208 to end).
logger.info(f'total={len(frames)} | train={len(train_frames)} | val={len(val_frames)}')

# ── STEP 4: CREATE YOLO FOLDERS ───────────────────────────────────────────────
banner("STEP 4 - CREATE YOLO FOLDERS")

# 4a. dataset_path = dataset/<dataset_name>
if not dataset_name:
    raise ValueError("dataset_name is empty in config.yaml, refusing to delete dataset/")
dataset_path = os.path.join("dataset", dataset_name)   # e.g dataset/5r_8c_9r_10r_11c_14c

# 4b. if it already exists -> delete it (fresh rebuild)
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path) # deletes the folder AND everything inside it
    logger.info(f"old dataset deleted: {dataset_path}")

# 4c. create images/train, images/val, labels/train, labels/val
for subfolder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(dataset_path, subfolder), exist_ok=True)
logger.info(f"YOLO folders created in {dataset_path}")

# ── STEP 5: COPY IMAGES + WRITE LABEL FILES ───────────────────────────────────
banner("STEP 5 - COPY IMAGES + WRITE LABEL FILES")

# 5a. copies the images + labels of ONE split (train or val); called twice below
def copy_split(split, split_frames): # e.g copy_split("train", train_frames)
    """For one split (train or val), it copies each photo into images/<split>/ and writes a matching .txt file of its MySQL boxes into labels/<split>/"""
    for frame_id, frame_path in split_frames:
        frame_name = os.path.basename(frame_path)   # e.g frame_1800_IMG_0909.jpg

        # 5b. copy the image into images/<split>/
        shutil.copy2(frame_path, os.path.join(dataset_path, "images", split, frame_name)) # shutil.copy2(FROM, TO)
        logger.debug(f"{split} | frame_id={frame_id} | image copied: {frame_name}")

        # 5c. SELECT this frame's bboxes, only from the chosen annotation sets
        reading_cursor.execute(
            f"SELECT class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s AND annotation_set_id IN ({placeholders})",
            (frame_id, *annotation_set_ids)) # without the * → (336, [5, 9, 10]) -> with the * unpacks to (336, 5, 9, 10)
        boxes = reading_cursor.fetchall()   # e.g [(0, 0.41, 0.55, 0.08, 0.04), ...]
        logger.debug(f"{split} | frame_id={frame_id} | {len(boxes)} boxes fetched")

        # 5d. write labels/<split>/<frame_name>.txt, one "class x y w h" line per box
        label_name = os.path.splitext(frame_name)[0] + ".txt"   # frame_1800_IMG_0909.jpg -> frame_1800_IMG_0909.txt
        with open(os.path.join(dataset_path, "labels", split, label_name), "w") as f:
            for class_id, x_center, y_center, width, height in boxes:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        logger.debug(f"{split} | frame_id={frame_id} | label written: {label_name} ({len(boxes)} lines)")

    logger.info(f"{split}: {len(split_frames)} images copied to {os.path.join(dataset_path, 'images', split)}")

copy_split("train", train_frames)
copy_split("val", val_frames)

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
