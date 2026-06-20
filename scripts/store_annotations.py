"""
Parses a LabelStudio YOLO export and stores bboxes in MySQL.

Input  : label .txt files from config labels_path
Needs  : frames already extracted and registered in MySQL
Output : one row in annotation_sets + rows in annotations table
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
import datetime
from pathlib import Path

import yaml

from scripts.db import get_connection, get_frame_id


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'

LABEL_MAP = {0: "danio_rerio", 1: "reflection"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_frame_number(label_file):
    """Extracts frame number from a LabelStudio filename: 'e6d83681-frame_360.txt' → 360."""
    stem = os.path.splitext(label_file.split("-")[1])[0]
    return int(stem.split("_")[1])


def parse_annotation_line(tokens):
    """Parses one YOLO bbox label line. Returns None if token count is unrecognised."""
    if len(tokens) == 5:
        return {
            'class_id': int(tokens[0]),
            'x_center': float(tokens[1]),
            'y_center': float(tokens[2]),
            'width':    float(tokens[3]),
            'height':   float(tokens[4]),
        }
    return None


def get_video_id(cursor, video_name):
    """Looks up video_id from videos table using video filename."""
    cursor.execute("SELECT id FROM videos WHERE file_path LIKE %s", (f"%{video_name}%",))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"video '{video_name}' not found in videos table — run sync_videos.py first")
    return row[0]


def read_sidecar(frames_folder):
    """Reads extraction_params.yaml from the frames folder if present."""
    path = os.path.join(frames_folder, 'extraction_params.yaml')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def create_annotation_set(cursor, video_id, frame_source, notes, frames_extracted,
                          iou_threshold, dedup_window, sample_rate, start_seconds, end_seconds):
    """Creates one annotation_sets row and returns its id."""
    cursor.execute(
        """INSERT INTO annotation_sets
           (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
            sample_rate, start_seconds, end_seconds, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
         sample_rate, start_seconds, end_seconds, datetime.datetime.now())
    )
    return cursor.lastrowid


def print_summary(annotation_set_id, labels_path, frames_folder, total_frames, total_annotations):
    print("\n" + "─" * 50)
    print("  SUMMARY")
    print("─" * 50)
    print(f"  annotation_set_id    : {annotation_set_id}")
    print(f"  Created at           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Labels path          : {labels_path}")
    print(f"  Frames folder        : {frames_folder}")
    print(f"  Frames processed     : {total_frames}")
    print(f"  Annotations inserted : {total_annotations}")
    print("─" * 50)
    print(f"\n  Cross-check in SQL:")
    print(f"  SELECT COUNT(*) FROM annotations WHERE annotation_set_id = {annotation_set_id};")
    print("─" * 50)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(conn=None, labels_path=None, frames_folder=None, video_name=None, frame_source=None, notes=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if labels_path is None:
        labels_path   = cfg['store_annotations']['labels_path']
        frames_folder = cfg['store_annotations']['frames_folder']
        video_name    = cfg['store_annotations']['video_name']
        frame_source  = cfg['store_annotations']['frame_source']
        notes         = cfg['store_annotations'].get('notes', '')

    logger.info(f"video_name={video_name}, frame_source={frame_source}, labels_path={labels_path}")

    total_frames      = 0
    total_annotations = 0

    if conn is None:
        conn = get_connection()

    reading_cursor = conn.cursor()
    insert_cursor  = conn.cursor()

    sidecar           = read_sidecar(frames_folder)
    frame_source      = sidecar.get('frame_source', frame_source)
    frames_extracted  = sidecar.get('frames_extracted')
    iou_threshold     = sidecar.get('iou_threshold')
    dedup_window      = sidecar.get('dedup_window')
    sample_rate       = sidecar.get('sample_rate')
    start_seconds     = sidecar.get('start_seconds')
    end_seconds       = sidecar.get('end_seconds')

    video_id          = get_video_id(reading_cursor, video_name)
    annotation_set_id = create_annotation_set(insert_cursor, video_id, frame_source, notes,
                                              frames_extracted, iou_threshold, dedup_window,
                                              sample_rate, start_seconds, end_seconds)
    logger.info(f"annotation_set_id={annotation_set_id}")

    for label_file in os.listdir(labels_path):

        frame_number = parse_frame_number(label_file)
        frame_id     = get_frame_id(reading_cursor, frames_folder, frame_number)
        total_frames += 1
        logger.debug(f"frame_number={frame_number} → frame_id={frame_id}")

        with open(os.path.join(labels_path, label_file)) as f:
            lines = f.readlines()

        for line in lines:
            ann = parse_annotation_line(line.split())
            if ann is None:
                logger.warning(f"{label_file} — unexpected token count. Skipping.")
                continue

            insert_cursor.execute(
                "INSERT INTO annotations (frame_id, annotation_set_id, class_id, label, x_center, y_center, width, height, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (frame_id, annotation_set_id, ann['class_id'], LABEL_MAP[ann['class_id']],
                 ann['x_center'], ann['y_center'], ann['width'], ann['height'],
                 datetime.datetime.now())
            )
            total_annotations += 1

    conn.commit()
    print_summary(annotation_set_id, labels_path, frames_folder, total_frames, total_annotations)


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
    bbox = parse_annotation_line(['0', '0.5', '0.4', '0.1', '0.08'])
    assert bbox['class_id'] == 0
    assert bbox['x_center'] == 0.5
    assert parse_annotation_line(['0', '0.5']) is None

def test_main(db_conn):
    print("\n*****************************************************")
    print("\n--- testing: main ---")
    main(db_conn, labels_path='fixtures/labels', frames_folder='frames/frames_IMG_0350_20260101_2000',
         video_name='IMG_0350', frame_source='1fps', notes='test import')
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations WHERE annotation_set_id = 2")
    assert cursor.fetchone()[0] == 1
