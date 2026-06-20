"""
Queries MySQL for annotated frames and builds a YOLO-ready dataset folder.

Input  : annotation_set_ids from config.yaml (integers from annotation_sets table)
Needs  : annotations stored in MySQL (run store_annotations.py first)
Output : dataset/{name}/images + labels folders, dataset_card.yaml, dataset.yaml
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import datetime
import logging
import os
import random
import shutil
import subprocess
from typing import Any

import mlflow
import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH  = 'config.yaml'
TRAIN_SPLIT  = 0.8
RANDOM_SEED  = 42


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fetch_annotated_frames(cursor: Any, annotation_set_ids: list[int]) -> list[tuple[int, str]]:
    placeholders = ','.join(['%s'] * len(annotation_set_ids))
    cursor.execute(
        f"""SELECT DISTINCT f.id, f.frame_path
            FROM annotations a
            JOIN frames f ON a.frame_id = f.id
            WHERE a.annotation_set_id IN ({placeholders})""",
        annotation_set_ids
    )
    return cursor.fetchall()


def split_train_val(frames: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    random.seed(RANDOM_SEED)
    random.shuffle(frames)
    split_at = int(len(frames) * TRAIN_SPLIT)
    return frames[:split_at], frames[split_at:]


def create_dataset_dirs(dataset_name: str) -> None:
    """Creates the YOLO folder structure — deletes existing dataset folder first if present."""
    dataset_path = f"dataset/{dataset_name}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(f"{dataset_path}/{subfolder}", exist_ok=True)
    logger.debug(f"dataset dirs created: {dataset_path}")


def generate_dataset(frames: list[tuple[int, str]], split: str, dataset_name: str, conn: Any, annotation_set_ids: list[int]) -> None:
    """Copies images and writes YOLO label files for one split (train or val)."""
    reading_cursor = conn.cursor()
    placeholders = ','.join(['%s'] * len(annotation_set_ids))
    for frame_id, frame_path in frames:
        frame_basename = os.path.basename(frame_path)
        logger.debug(f"{split} → frame_id={frame_id}")

        shutil.copy2(frame_path, f"dataset/{dataset_name}/images/{split}/{frame_basename}")

        reading_cursor.execute(
            f"SELECT class_id, x_center, y_center, width, height FROM annotations "
            f"WHERE frame_id = %s AND annotation_set_id IN ({placeholders})",
            (frame_id, *annotation_set_ids)
        )
        annotations = reading_cursor.fetchall()

        txt_file = f"dataset/{dataset_name}/labels/{split}/{os.path.splitext(frame_basename)[0]}.txt"
        with open(txt_file, 'w') as f:
            for class_id, x_center, y_center, width, height in annotations:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        logger.debug(f"label file → {txt_file}")


def fetch_video_metadata(cursor: Any, annotation_set_ids: list[int]) -> list[tuple]:
    placeholders = ','.join(['%s'] * len(annotation_set_ids))
    cursor.execute(
        f"""SELECT DISTINCT v.file_path, v.species, v.morph,
                   v.tank_width_cm, v.tank_height_cm, v.tank_depth_cm,
                   v.fps, v.resolution, v.fish_count, v.notes,
                   CAST(v.filmed_at AS CHAR) AS filmed_at
            FROM videos v
            JOIN frames f ON f.video_id = v.id
            JOIN annotations a ON a.frame_id = f.id
            WHERE a.annotation_set_id IN ({placeholders})""",
        annotation_set_ids
    )
    return cursor.fetchall()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def write_dataset_card(dataset_name: str, annotation_set_ids: list[int], videos_meta: list, n_train: int, n_val: int, git_commit: str) -> None:
    card = {
        'dataset_name':       dataset_name,
        'annotation_set_ids': annotation_set_ids,
        'git_commit':         git_commit,
        'videos':             videos_meta,
        'num_train':          n_train,
        'num_val':            n_val,
        'total_frames':       n_train + n_val,
        'classes':            {0: 'danio_rerio', 1: 'reflection'},
        'split':              '80/20 train/val',
        'random_seed':        RANDOM_SEED,
        'created_at':         str(datetime.datetime.now()),
    }
    path = f"dataset/{dataset_name}/dataset_card.yaml"
    with open(path, 'w') as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset card → {path}")


def write_yolo_yaml(dataset_name: str) -> None:
    dataset_yaml = {
        'path':  f"dataset/{dataset_name}",
        'train': 'images/train',
        'val':   'images/val',
        'nc':    2,
        'names': ['danio_rerio', 'reflection'],
    }
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset.yaml updated → {dataset_name}")


def log_to_mlflow(dataset_name: str, annotation_set_ids: list[int], git_commit: str, n_train: int, n_val: int) -> None:
    mlflow.set_experiment("prepare_dataset")
    with mlflow.start_run(run_name=dataset_name):
        mlflow.log_params({
            'dataset_name':       dataset_name,
            'annotation_set_ids': str(annotation_set_ids),
            'git_commit':         git_commit,
            'train_split':        TRAIN_SPLIT,
            'random_seed':        RANDOM_SEED,
        })
        mlflow.log_metrics({
            'num_train': n_train,
            'num_val':   n_val,
            'total':     n_train + n_val,
        })
    logger.info(f"MLflow run logged — dataset={dataset_name}, annotation_set_ids={annotation_set_ids}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(conn: Any = None) -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    annotation_set_ids = cfg['prepare_dataset']['annotation_set_ids']
    dataset_name       = cfg['prepare_dataset']['dataset_name']

    logger.info(f"annotation_set_ids={annotation_set_ids}, dataset_name={dataset_name}")

    if conn is None:
        conn = get_connection()

    reading_cursor = conn.cursor()

    frames                   = fetch_annotated_frames(reading_cursor, annotation_set_ids)
    train_frames, val_frames = split_train_val(frames)

    logger.info(f"total={len(frames)} | train={len(train_frames)} | val={len(val_frames)}")

    create_dataset_dirs(dataset_name)
    generate_dataset(train_frames, 'train', dataset_name, conn, annotation_set_ids)
    generate_dataset(val_frames,   'val',   dataset_name, conn, annotation_set_ids)

    meta_cursor = conn.cursor(dictionary=True)
    videos_meta = fetch_video_metadata(meta_cursor, annotation_set_ids)
    git_commit  = get_git_commit()

    write_dataset_card(dataset_name, annotation_set_ids, videos_meta, len(train_frames), len(val_frames), git_commit)
    write_yolo_yaml(dataset_name)
    log_to_mlflow(dataset_name, annotation_set_ids, git_commit, len(train_frames), len(val_frames))

    print("\n" + "─" * 50)
    print("  DATASET SUMMARY")
    print("─" * 50)
    print(f"  Dataset              : {dataset_name}")
    print(f"  Annotation set IDs   : {annotation_set_ids}")
    print(f"  Git commit           : {git_commit}")
    print(f"  Total frames         : {len(frames)}")
    print(f"  Train                : {len(train_frames)}")
    print(f"  Val                  : {len(val_frames)}")
    print(f"  Output               : dataset/{dataset_name}")
    print("─" * 50)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()


