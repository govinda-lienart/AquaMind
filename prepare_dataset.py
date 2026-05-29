import mysql.connector
import os
import random
import shutil
import yaml

from db import get_connection
conn = get_connection()

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

dataset_name   = cfg['prepare_dataset']['dataset_name']
frames_folders = cfg['prepare_dataset']['frames_folders']

reading_cursor = conn.cursor()

# ── COLLECT ALL ANNOTATED FRAMES FROM ALL VIDEOS ──────────────────────────────
labeled_frames = []
for frames_folder in frames_folders:
    frame_path_pattern = f"{frames_folder}%"
    reading_cursor.execute(
        'SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id WHERE f.frame_path LIKE %s',
        (frame_path_pattern,)
    )
    rows = reading_cursor.fetchall()
    print(f"{frames_folder} → {len(rows)} annotated frames")
    labeled_frames.extend(rows)

print(f"\nTotal annotated frames across all videos: {len(labeled_frames)}")

# ── SHUFFLE AND SPLIT 80/20 ───────────────────────────────────────────────────
random.seed(42)
random.shuffle(labeled_frames)

training_frames_80p = int(len(labeled_frames) * 0.8)
select_training   = labeled_frames[0:training_frames_80p]
select_validation = labeled_frames[training_frames_80p:]

print(f"Train: {len(select_training)} frames")
print(f"Val:   {len(select_validation)} frames")

# ── CREATE DATASET FOLDER ─────────────────────────────────────────────────────
if os.path.exists(f"dataset/{dataset_name}"):
    shutil.rmtree(f"dataset/{dataset_name}")

os.makedirs(f"dataset/{dataset_name}/images/train/", exist_ok=True)
os.makedirs(f"dataset/{dataset_name}/images/val/",   exist_ok=True)
os.makedirs(f"dataset/{dataset_name}/labels/train",  exist_ok=True)
os.makedirs(f"dataset/{dataset_name}/labels/val",    exist_ok=True)

# ── GENERATE DATASET ──────────────────────────────────────────────────────────
def generate_dataset(select_frames, split, dataset_name, conn):
    reading_cursor = conn.cursor()
    for frame_id, frame_path in select_frames:
        frame_basename_png = os.path.basename(frame_path)

        print(f'\n {split} -> Frame_id: {frame_id}')
        os.symlink(os.path.abspath(frame_path), f"dataset/{dataset_name}/images/{split}/{frame_basename_png}")
        print(f"-Symlink -> {dataset_name}/images/{split}/{frame_basename_png}")

        reading_cursor.execute(
            "SELECT class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s",
            (frame_id,))
        annotations = reading_cursor.fetchall()
        print(f"-Annotations -> {annotations}")

        frame_basename = os.path.splitext(frame_basename_png)[0]
        txt_file = f"dataset/{dataset_name}/labels/{split}/{frame_basename}.txt"

        with open(txt_file, "w") as f:
            for row in annotations:
                class_id, x_center, y_center, width, height = row
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"-Label file -> {txt_file}")

generate_dataset(select_training,   "train", dataset_name, conn)
generate_dataset(select_validation, "val",   dataset_name, conn)

# ── WRITE DATASET CARD ────────────────────────────────────────────────────────
card = {
    'dataset_name':    dataset_name,
    'frames_folders':  frames_folders,
    'num_train':       len(select_training),
    'num_val':         len(select_validation),
    'total_frames':    len(labeled_frames),
    'classes':         {0: 'danio_rerio', 1: 'reflection'},
    'split':           '80/20 train/val',
    'random_seed':     42,
    'created_at':      str(__import__('datetime').datetime.now()),
}

with open(f"dataset/{dataset_name}/dataset_card.yaml", "w") as f:
    yaml.dump(card, f, default_flow_style=False, sort_keys=False)

print(f"Dataset card written to dataset/{dataset_name}/dataset_card.yaml")

# ── UPDATE DATASET.YAML FOR YOLO TRAINING ────────────────────────────────────
dataset_yaml = {
    'path':  os.path.abspath(f"dataset/{dataset_name}"),
    'train': 'images/train',
    'val':   'images/val',
    'nc':    2,
    'names': ['danio_rerio', 'reflection'],
}

with open("dataset.yaml", "w") as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

print(f"dataset.yaml updated → {dataset_name}")
print(f"\n\033[92mDataset generated - check dataset/{dataset_name}\033[0m")
