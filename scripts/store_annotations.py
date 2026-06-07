# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import datetime

import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'

LABEL_MAP = {0: "danio_rerio", 1: "reflection"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_frame_number(label_file):
    stem = os.path.splitext(label_file.split("-")[1])[0]  # 'e6d83681-frame_360.txt' → 'frame_360'
    return int(stem.split("_")[1])                         # 'frame_360' → 360


def get_frame_id(cursor, frames_folder, frame_number):
    pattern = f"{frames_folder}/frame_{frame_number}%.png"
    cursor.execute("SELECT id FROM frames WHERE frame_path LIKE %s", (pattern,))
    row = cursor.fetchone()
    cursor.fetchall()  # drain cursor before reusing
    if row is None:
        raise ValueError(f"No DB record found for frame_{frame_number} in {frames_folder}")
    return row[0]


def parse_annotation_line(tokens):
    if len(tokens) == 5:
        return {
            'class_id':     int(tokens[0]),
            'x_center':     float(tokens[1]),
            'y_center':     float(tokens[2]),
            'width':        float(tokens[3]),
            'height':       float(tokens[4]),
            'has_keypoint': False,
        }
    if len(tokens) == 8:
        return {
            'class_id':     int(tokens[0]),
            'x_center':     float(tokens[1]),
            'y_center':     float(tokens[2]),
            'width':        float(tokens[3]),
            'height':       float(tokens[4]),
            'kp_x':         float(tokens[5]),
            'kp_y':         float(tokens[6]),
            'kp_visible':   int(tokens[7]),   # visibility flag: 2 = visible, 0 = not visible
            'has_keypoint': True,
        }
    return None


def print_summary(session_id, labels_path, frames_folder, total_frames, total_annotations, total_keypoints):
    print("\n" + "─" * 50)
    print("  SUMMARY")
    print("─" * 50)
    print(f"  Session ID       : {session_id}")
    print(f"  Created at       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Labels path      : {labels_path}")
    print(f"  Frames folder    : {frames_folder}")
    print(f"  Frames processed     : {total_frames}")
    print(f"  Annotations inserted : {total_annotations}")
    print(f"  Keypoints inserted   : {total_keypoints}")
    print("─" * 50)
    print(f"\n  Cross-check in SQL:")
    print(f"  SELECT COUNT(*) FROM annotations WHERE session_id = '{session_id}';")
    print(f"  SELECT COUNT(*) FROM keypoints k JOIN annotations a ON a.id = k.annotation_id WHERE a.session_id = '{session_id}';")
    print("─" * 50)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    labels_path   = cfg['store_annotations']['labels_path']
    frames_folder = cfg['store_annotations']['frames_folder']
    session_id    = 'annot_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')

    print(f"Session ID: {session_id}")

    total_frames      = 0
    total_annotations = 0
    total_keypoints   = 0

    with get_connection() as conn:
        reading_cursor = conn.cursor()
        insert_cursor  = conn.cursor()

        for label_file in os.listdir(labels_path):

            frame_number = parse_frame_number(label_file)
            frame_id     = get_frame_id(reading_cursor, frames_folder, frame_number)
            total_frames += 1
            print(f"\nframe number: {frame_number} → frame_id: {frame_id}")

            annotation_path = os.path.join(labels_path, label_file)
            with open(annotation_path) as f:
                lines = f.readlines()

            for line in lines:
                tokens = line.split()
                ann    = parse_annotation_line(tokens)

                if ann is None:
                    print(f"WARNING: {label_file} — {len(tokens)} tokens, expected 5 or 8. Skipping.")
                    continue

                label      = LABEL_MAP[ann['class_id']]
                created_at = datetime.datetime.now()

                insert_cursor.execute(
                    "INSERT INTO annotations (frame_id, class_id, label, x_center, y_center, width, height, created_at, session_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (frame_id, ann['class_id'], label, ann['x_center'], ann['y_center'], ann['width'], ann['height'], created_at, session_id)
                )
                annotation_id  = insert_cursor.lastrowid
                total_annotations += 1

                if ann['has_keypoint'] and label == 'danio_rerio':
                    insert_cursor.execute(
                        "INSERT INTO keypoints (annotation_id, name, x, y, visible, created_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (annotation_id, 'eye', ann['kp_x'], ann['kp_y'], ann['kp_visible'], created_at)
                    )
                    total_keypoints += 1
                    print(f"  keypoint eye → ({ann['kp_x']:.4f}, {ann['kp_y']:.4f}) visible={ann['kp_visible']}")

        conn.commit()

    print_summary(session_id, labels_path, frames_folder, total_frames, total_annotations, total_keypoints)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
