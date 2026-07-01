"""
Parses a LabelStudio YOLO export and stores bboxes in MySQL.

Input  : label .txt files from config labels_path
Needs  : frames already extracted and registered in MySQL
Output : one row in annotation_sets + rows in annotations table (bboxes)
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
    """Extracts frame number from a LabelStudio filename: """
    stem = os.path.splitext(label_file.split("-")[1])[0] # os.path.splitext("frame_360.txt")  →  ("frame_360", ".txt")
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



def read_sidecar(path: str) -> dict[str, Any]:
    """Reads a YAML sidecar file if present, returns empty dict if not found."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def create_annotation_set(cursor: Any, video_id: int, frame_source: str, notes: str | None,
                          frames_extracted: int | None, iou_threshold: float | None,
                          dedup_window: int | None, sample_rate: int | None,
                          start_seconds: float | None, end_seconds: float | None,
                          ls_project_name: str | None, ls_project_id: int | None,
                          ls_min_task_id: int | None, ls_max_task_id: int | None,
                          ls_downloaded_at: str | None) -> int:
    """Creates one annotation_sets row and returns its id."""
    cursor.execute(
        """INSERT INTO annotation_sets
           (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
            sample_rate, start_seconds, end_seconds, created_at,
            ls_project_name, ls_project_id, ls_min_task_id, ls_max_task_id, ls_downloaded_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (video_id, frame_source, notes, frames_extracted, iou_threshold, dedup_window,
         sample_rate, start_seconds, end_seconds, datetime.datetime.now(),
         ls_project_name, ls_project_id, ls_min_task_id, ls_max_task_id, ls_downloaded_at)
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

def main(conn: MySQLConnection | None = None) -> None: # input will be None in current pipeline - scripts called directly

    # Connects with database
    if conn is None:
        conn = get_connection()
        logger.info("connected to sql database")

    # Establishing sql cursors

    reading_cursor = conn.cursor() # SELECT queries (fetching)
    insert_cursor  = conn.cursor() # INSERT in mysql

    # Read and Load config values from config.yaml

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)['store_annotations'] # dictionary
    labels_path   = cfg['labels_path'] # labelstudio_download/labelstudio_IMG_0764_25062026_14h00/labels
    frames_folder = cfg['frames_folder'] # frames/frames_IMG_0764_20260624_1635
    video_name    = cfg['video_name']  #"IMG_0350.MOV"
    frame_source  = cfg['frame_source'] # regular or crossing
    notes         = cfg.get('notes', '')

    logger.info(f"video_name={video_name}, frame_source={frame_source}, labels_path={labels_path}")



    total_frames      = 0
    total_annotations = 0

    # SIDE CAR READING - loading metadata from sidecar generated with extract_frames.py/extract_crossing_frames and labelstudio

    sidecar           = read_sidecar(os.path.join(frames_folder, 'extraction_params.yaml'))
    frame_source      = sidecar.get('frame_source', frame_source) # both script generate either regular or crossing
    frames_extracted  = sidecar.get('frames_extracted') # boths scripts (number of extracted frames)
    iou_threshold     = sidecar.get('iou_threshold') # extract_crossing_frames
    dedup_window      = sidecar.get('dedup_window') # extract_crossing_frames
    sample_rate       = sidecar.get('sample_rate') # extract_crossing_frames
    start_seconds     = sidecar.get('start_seconds') # boths scripts (start video)
    end_seconds       = sidecar.get('end_seconds') # boths scripts (end video)

    dl_sidecar        = read_sidecar(os.path.join(os.path.dirname(labels_path), 'download_params.yaml')) # labelstudio
    ls_project_name   = dl_sidecar.get('project_name') # name of the project in labelstudio
    ls_project_id     = dl_sidecar.get('project_id') # project id in ls
    ls_min_task_id    = dl_sidecar.get('min_task_id') # selection of the start id in ls
    ls_max_task_id    = dl_sidecar.get('max_task_id') # select end id in ls
    ls_downloaded_at  = dl_sidecar.get('downloaded_at') # date downloaded from ls

    # write annotation_set_id

    video_id          = get_video_id(reading_cursor, video_name) # video id is used in anno_set_id as FK
    annotation_set_id = create_annotation_set(insert_cursor, video_id, frame_source, notes,
                                              frames_extracted, iou_threshold, dedup_window,
                                              sample_rate, start_seconds, end_seconds,
                                              ls_project_name, ls_project_id,
                                              ls_min_task_id, ls_max_task_id, ls_downloaded_at)
    logger.info(f"annotation_set_id={annotation_set_id}")

    # processing every annotation file from labelstudio and storing the bboxes in MySQL


    total_frames      = 0
    total_annotations = 0

    for label_file in os.listdir(labels_path): # Loop over .txt files in the labels folder — one file per frame
        if not label_file.endswith('.txt'):
            continue

        frame_number = parse_frame_number(label_file) # eg 0d7adc63-crossing_000824.txt -> 000824
        try:
            frame_id = get_frame_id(reading_cursor, frames_folder, frame_number) # queries frames table uising frames folder and frames number to finding matching row --> returns id PK of frames
        except ValueError:
            logger.warning(f"frame {frame_number} not in MySQL — skipping")
            continue

        total_frames += 1
        logger.debug(f"frame_number={frame_number} → frame_id={frame_id}")

        with open(os.path.join(labels_path, label_file)) as f: #s
            lines = f.readlines() # reading bboxes data

        for line in lines: # read line per line in each txt file
            ann = parse_annotation_line(line.split()[:5]) # "0 0.512 0.341 0.089 0.045"  →  {'class_id': 0, 'x_center': 0.512, ...} 
            if ann is None:
                logger.warning(f"{label_file} — unexpected token count. Skipping.") #
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

