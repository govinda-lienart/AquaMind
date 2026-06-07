# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import random
import shutil
import datetime

import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42


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
    dataset_path = f"dataset/{dataset_name}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(f"{dataset_path}/{subfolder}", exist_ok=True)


def generate_dataset(frames, split, dataset_name, conn, mode):
    reading_cursor = conn.cursor()
    kp_cursor      = conn.cursor()
    for frame_id, frame_path in frames:
        frame_basename_png = os.path.basename(frame_path)
        print(f'\n{split} → frame_id: {frame_id}')

        os.symlink(os.path.abspath(frame_path), f"dataset/{dataset_name}/images/{split}/{frame_basename_png}")
        print(f"  symlink → {dataset_name}/images/{split}/{frame_basename_png}")

        reading_cursor.execute(
            "SELECT id, class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s",
            (frame_id,)
        )
        annotations = reading_cursor.fetchall()
        print(f"  annotations → {annotations}")

        frame_basename = os.path.splitext(frame_basename_png)[0]
        txt_file       = f"dataset/{dataset_name}/labels/{split}/{frame_basename}.txt"

        with open(txt_file, 'w') as f:
            for annotation_id, class_id, x_center, y_center, width, height in annotations:
                if mode == 'pose' and class_id == 0:
                    kp_cursor.execute(
                        "SELECT x, y, visible FROM keypoints WHERE annotation_id = %s AND name = 'eye'",
                        (annotation_id,)
                    )
                    kp = kp_cursor.fetchone()
                    kp_x, kp_y, kp_v = kp if kp else (0.0, 0.0, 0)
                    f.write(f"{class_id} {x_center} {y_center} {width} {height} {kp_x} {kp_y} {kp_v}\n")
                else:
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"  label file → {txt_file}")


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
    print(f"Dataset card written to {path}")


def write_yolo_yaml(dataset_name, mode):
    dataset_yaml = {
        'path':  f"dataset/{dataset_name}",
        'train': 'images/train',
        'val':   'images/val',
        'nc':    2,
        'names': ['danio_rerio', 'reflection'],
    }
    if mode == 'pose':
        dataset_yaml['kpt_shape'] = [1, 3]  # 1 keypoint (eye), 3 values (x, y, visible)
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"dataset.yaml updated → {dataset_name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sessions     = cfg['prepare_dataset']['annotation_sessions']
    dataset_name = cfg['prepare_dataset']['dataset_name']
    mode         = cfg['prepare_dataset'].get('mode', 'bbox')

    print(f"Building dataset from sessions: {sessions}")
    print(f"Mode: {mode}")

    with get_connection() as conn:
        reading_cursor = conn.cursor()

        frames                   = fetch_annotated_frames(reading_cursor, sessions)
        train_frames, val_frames = split_train_val(frames)

        print(f"Total annotated frames : {len(frames)}")
        print(f"Train: {len(train_frames)} | Val: {len(val_frames)}")

        create_dataset_dirs(dataset_name)

        generate_dataset(train_frames, 'train', dataset_name, conn, mode)
        generate_dataset(val_frames,   'val',   dataset_name, conn, mode)

        meta_cursor = conn.cursor(dictionary=True)
        videos_meta = fetch_video_metadata(meta_cursor, sessions)

    write_dataset_card(dataset_name, mode, sessions, videos_meta, len(train_frames), len(val_frames))
    write_yolo_yaml(dataset_name, mode)

    print(f"\033[92mDataset generated — check dataset/{dataset_name}\033[0m")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
