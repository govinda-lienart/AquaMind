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
from typing import Any
import yaml

from scripts.db import get_connection, get_frame_id, get_video_id


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'

LABEL_MAP = {0: "danio_rerio", 1: "reflection"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_frame_number(label_file: str) -> int:
    """Extracts frame number from a LabelStudio filename: 'e6d83681-frame_360.txt' → 360."""
    stem = os.path.splitext(label_file.split("-")[1])[0]
    return int(stem.split("_")[1])


def parse_annotation_line(tokens: list[str]) -> dict[str, Any] | None:
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



def read_sidecar(frames_folder: str) -> dict[str, Any]:
    """Reads extraction_params.yaml from the frames folder if present."""
    path = os.path.join(frames_folder, 'extraction_params.yaml')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def create_annotation_set(cursor: Any, video_id: int, frame_source: str, notes: str | None,
                          frames_extracted: int | None, iou_threshold: float | None,
                          dedup_window: int | None, sample_rate: int | None,
                          start_seconds: float | None, end_seconds: float | None) -> int:
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


def print_summary(annotation_set_id: int, labels_path: str, frames_folder: str, total_frames: int, total_annotations: int) -> None:
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

def main(conn: Any = None, labels_path: str | None = None, frames_folder: str | None = None, video_name: str | None = None, frame_source: str | None = None, notes: str | None = None) -> None:
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
        if not label_file.endswith('.txt'):
            continue

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

