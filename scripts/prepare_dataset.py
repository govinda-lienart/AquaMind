"""
Queries MySQL for annotated frames and builds a YOLO-ready dataset folder.

Input  : annotation session IDs from config.yaml
Needs  : annotations stored in MySQL (run store_annotations.py first)
Output : dataset/{name}/images + labels folders, dataset_card.yaml, dataset.yaml
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import datetime
import logging
import os
import random
import shutil

import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH  = 'config.yaml'
TRAIN_SPLIT  = 0.8
RANDOM_SEED  = 42


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fetch_annotated_frames(cursor, sessions):
    placeholders = ','.join(['%s'] * len(sessions))
    cursor.execute(
        f"SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id WHERE a.session_id IN ({placeholders})",
        sessions
    )
    return cursor.fetchall()


def split_train_val(frames):
    random.seed(RANDOM_SEED)
    random.shuffle(frames)
    split_at = int(len(frames) * TRAIN_SPLIT)
    return frames[:split_at], frames[split_at:]


def create_dataset_dirs(dataset_name):
    """Creates the YOLO folder structure — deletes existing dataset folder first if present."""
    dataset_path = f"dataset/{dataset_name}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(f"{dataset_path}/{subfolder}", exist_ok=True)
    logger.debug(f"dataset dirs created: {dataset_path}")


def generate_dataset(frames, split, dataset_name, conn, mode):
    """Copies images and writes YOLO label files for one split (train or val)."""
    reading_cursor = conn.cursor()
    kp_cursor      = conn.cursor()
    for frame_id, frame_path in frames:
        frame_basename_png = os.path.basename(frame_path)
        logger.debug(f"{split} → frame_id={frame_id}")

        shutil.copy2(frame_path, f"dataset/{dataset_name}/images/{split}/{frame_basename_png}")
        logger.debug(f"copy → {dataset_name}/images/{split}/{frame_basename_png}")

        reading_cursor.execute(
            "SELECT id, class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s",
            (frame_id,)
        )
        annotations = reading_cursor.fetchall()
        logger.debug(f"annotations → {annotations}")

        frame_basename = os.path.splitext(frame_basename_png)[0]
        txt_file       = f"dataset/{dataset_name}/labels/{split}/{frame_basename}.txt"

        with open(txt_file, 'w') as f:
            for annotation_id, class_id, x_center, y_center, width, height in annotations:
                if mode == 'pose':
                    if class_id == 0:
                        kp_cursor.execute(
                            "SELECT x, y, visible FROM keypoints WHERE annotation_id = %s AND name = 'eye'",
                            (annotation_id,)
                        )
                        kp = kp_cursor.fetchone()
                        kp_x, kp_y, kp_v = kp if kp else (0.0, 0.0, 0)
                    else:
                        kp_x, kp_y, kp_v = 0.0, 0.0, 0
                    f.write(f"{class_id} {x_center} {y_center} {width} {height} {kp_x} {kp_y} {kp_v}\n")
                else:
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        logger.debug(f"label file → {txt_file}")


def fetch_video_metadata(cursor, sessions):
    placeholders = ','.join(['%s'] * len(sessions))
    cursor.execute(
        f"""SELECT DISTINCT v.file_path, v.species, v.morph,
                   v.tank_width_cm, v.tank_height_cm, v.tank_depth_cm,
                   v.fps, v.resolution, v.fish_count, v.notes,
                   CAST(v.filmed_at AS CHAR) AS filmed_at
            FROM videos v
            JOIN frames f ON f.video_id = v.id
            JOIN annotations a ON a.frame_id = f.id
            WHERE a.session_id IN ({placeholders})""",
        sessions
    )
    return cursor.fetchall()


def write_dataset_card(dataset_name, mode, sessions, videos_meta, n_train, n_val):
    card = {
        'dataset_name':        dataset_name,
        'mode':                mode,
        'annotation_sessions': sessions,
        'videos':              videos_meta,
        'num_train':           n_train,
        'num_val':             n_val,
        'total_frames':        n_train + n_val,
        'classes':             {0: 'danio_rerio', 1: 'reflection'},
        'split':               '80/20 train/val',
        'random_seed':         RANDOM_SEED,
        'created_at':          str(datetime.datetime.now()),
    }
    path = f"dataset/{dataset_name}/dataset_card.yaml"
    with open(path, 'w') as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset card → {path}")


def write_yolo_yaml(dataset_name, mode):
    dataset_yaml = {
        'path':  f"dataset/{dataset_name}",
        'train': 'images/train',
        'val':   'images/val',
        'nc':    2,
        'names': ['danio_rerio', 'reflection'],
    }
    if mode == 'pose':
        dataset_yaml['kpt_shape'] = [1, 3]
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset.yaml updated → {dataset_name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(conn=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sessions     = cfg['prepare_dataset']['annotation_sessions']
    dataset_name = cfg['prepare_dataset']['dataset_name']
    mode         = cfg['prepare_dataset'].get('mode', 'bbox')

    logger.info(f"sessions={sessions}, dataset_name={dataset_name}, mode={mode}")

    if conn is None:
        conn = get_connection()

    reading_cursor = conn.cursor()

    frames                   = fetch_annotated_frames(reading_cursor, sessions)
    train_frames, val_frames = split_train_val(frames)

    logger.info(f"total={len(frames)} | train={len(train_frames)} | val={len(val_frames)}")

    create_dataset_dirs(dataset_name)

    generate_dataset(train_frames, 'train', dataset_name, conn, mode)
    generate_dataset(val_frames,   'val',   dataset_name, conn, mode)

    meta_cursor = conn.cursor(dictionary=True)
    videos_meta = fetch_video_metadata(meta_cursor, sessions)

    write_dataset_card(dataset_name, mode, sessions, videos_meta, len(train_frames), len(val_frames))
    write_yolo_yaml(dataset_name, mode)

    print("\n" + "─" * 50)
    print("  DATASET SUMMARY")
    print("─" * 50)
    print(f"  Dataset              : {dataset_name}")
    print(f"  Mode                 : {mode}")
    print(f"  Sessions             : {len(sessions)}")
    for s in sessions:
        print(f"    - {s}")
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


# ── TESTS ─────────────────────────────────────────────────────────────────────
#  pytest scripts/prepare_dataset.py -v -s

def test_split_train_val():
    print("\n*****************************************************")
    print("\n--- testing: split_train_val ---")
    frames = [(i, f'frame_{i}.png') for i in range(10)]
    train, val = split_train_val(frames)
    assert len(train) == 8
    assert len(val)   == 2
    assert len(train) + len(val) == len(frames)

def test_create_dataset_dirs(tmp_path, monkeypatch):
    print("\n*****************************************************")
    print("\n--- testing: create_dataset_dirs ---")
    monkeypatch.chdir(tmp_path)
    create_dataset_dirs('test_ds')
    assert os.path.isdir('dataset/test_ds/images/train')
    assert os.path.isdir('dataset/test_ds/images/val')
    assert os.path.isdir('dataset/test_ds/labels/train')
    assert os.path.isdir('dataset/test_ds/labels/val')

def test_write_dataset_card(tmp_path, monkeypatch):
    print("\n*****************************************************")
    print("\n--- testing: write_dataset_card ---")
    monkeypatch.chdir(tmp_path)
    os.makedirs('dataset/test_ds')
    write_dataset_card('test_ds', 'bbox', ['annot_test'], [], n_train=8, n_val=2)
    with open('dataset/test_ds/dataset_card.yaml') as f:
        card = yaml.safe_load(f)
    assert card['dataset_name'] == 'test_ds'
    assert card['num_train']    == 8
    assert card['num_val']      == 2
    assert card['mode']         == 'bbox'

def test_write_yolo_yaml(tmp_path, monkeypatch):
    print("\n*****************************************************")
    print("\n--- testing: write_yolo_yaml ---")
    monkeypatch.chdir(tmp_path)
    write_yolo_yaml('test_ds', 'bbox')
    with open('dataset.yaml') as f:
        data = yaml.safe_load(f)
    assert data['nc']    == 2
    assert data['names'] == ['danio_rerio', 'reflection']

def test_write_yolo_yaml_pose(tmp_path, monkeypatch):
    print("\n*****************************************************")
    print("\n--- testing: write_yolo_yaml pose mode ---")
    monkeypatch.chdir(tmp_path)
    write_yolo_yaml('test_ds', 'pose')
    with open('dataset.yaml') as f:
        data = yaml.safe_load(f)
    assert data['kpt_shape'] == [1, 3]

def test_main(db_conn, tmp_path, monkeypatch):
    print("\n*****************************************************")
    print("\n--- testing: main ---")
    shutil.copy('config.yaml', tmp_path)
    monkeypatch.chdir(tmp_path)
    main(db_conn)  # sessions from config won't match aquamind_test → 0 frames, pipeline still runs
