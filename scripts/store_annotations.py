"""
Parses a LabelStudio YOLO export and stores bboxes and eye keypoints in MySQL.

Input  : label .txt files from config labels_path
Needs  : frames already extracted and registered in MySQL
Output : rows inserted into annotations and keypoints tables
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
import datetime

import yaml

from scripts.db import get_connection, get_frame_id


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'

LABEL_MAP = {0: "danio_rerio", 1: "reflection"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_frame_number(label_file):
    """Extracts frame number from a LabelStudio filename: 'e6d83681-frame_360.txt' → 360."""
    stem = os.path.splitext(label_file.split("-")[1])[0]  # 'e6d83681-frame_360.txt' → 'frame_360'
    return int(stem.split("_")[1])                         # 'frame_360' → 360


def parse_annotation_line(tokens):
    """Parses one YOLO label line into a dict. Returns None if token count is unrecognised."""
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

def main(conn=None, labels_path=None, frames_folder=None):
    if labels_path is None:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        labels_path   = cfg['store_annotations']['labels_path']
        frames_folder = cfg['store_annotations']['frames_folder']

    session_id = 'annot_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    logger.info(f"session_id={session_id}, labels_path={labels_path}, frames_folder={frames_folder}")

    total_frames      = 0
    total_annotations = 0
    total_keypoints   = 0

    if conn is None:
        conn = get_connection()

    reading_cursor = conn.cursor()
    insert_cursor  = conn.cursor()

    for label_file in os.listdir(labels_path):

        frame_number = parse_frame_number(label_file)
        frame_id     = get_frame_id(reading_cursor, frames_folder, frame_number)
        total_frames += 1
        logger.debug(f"frame_number={frame_number} → frame_id={frame_id}")

        annotation_path = os.path.join(labels_path, label_file)
        with open(annotation_path) as f:
            lines = f.readlines()

        for line in lines:
            tokens = line.split()
            ann    = parse_annotation_line(tokens)

            if ann is None:
                logger.warning(f"{label_file} — {len(tokens)} tokens, expected 5 or 8. Skipping.")
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
                logger.debug(f"keypoint eye → ({ann['kp_x']:.4f}, {ann['kp_y']:.4f}) visible={ann['kp_visible']}")

    conn.commit()

    print_summary(session_id, labels_path, frames_folder, total_frames, total_annotations, total_keypoints)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()

# ── TESTS ─────────────────────────────────────────────────────────────────────
#  pytest scripts/store_annotations.py -v -s

def test_parse_frame_number():
    print("\n*****************************************************")
    print("\n--- testing: parse_frame_number ---")
    assert parse_frame_number('e6d83681-frame_360.txt') == 360
    assert parse_frame_number('abc12345-frame_0.txt')   == 0

def test_parse_annotation_line():
    print("\n*****************************************************")
    print("\n--- testing: parse_annotation_line ---")
    bbox_only = parse_annotation_line(['0', '0.5', '0.4', '0.1', '0.08'])
    assert bbox_only['class_id']     == 0
    assert bbox_only['has_keypoint'] == False

    with_keypoint = parse_annotation_line(['0', '0.5', '0.4', '0.1', '0.08', '0.31', '0.58', '2'])
    assert with_keypoint['has_keypoint'] == True
    assert with_keypoint['kp_x']         == 0.31

    assert parse_annotation_line(['0', '0.5']) is None  # unknown token count → None

def test_main():
    print("\n*****************************************************")
    print("\n--- testing: main ---")
    from scripts.db import aquatest_connection
    conn = aquatest_connection()
    try:
        main(conn, labels_path='fixtures/labels', frames_folder='frames/frames_IMG_0350_20260101_2000')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM annotations")
        assert cursor.fetchone()[0] == 2  # fixtures/labels has 1 file with 2 annotation lines
    finally:
        conn.close()
