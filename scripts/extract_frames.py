"""
Extracts/stores one frame per second from a video and stores the frame paths in MySQL.

Input  : video file registered in the videos table (run sync_videos.py first)
Output : PNG files on disk + rows inserted into the frames table
Guards : unique constrain5t (no duplicates), video must be registered, skips if already extracted
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import datetime
import logging
import re
from typing import Any
import yaml
import os
import cv2
from scripts.db import get_connection, get_video_id, register_frames

# ── LOGGER ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'

# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_path_storage_frames(video_path: str, frames_dir: str) -> str:
    """ Bulding path storage from loadad data config file"""
    logger.debug(f"video_path={video_path}, frames_dir={frames_dir}")
    video_name        = os.path.splitext(os.path.basename(video_path))[0]  # 'IMG_0909'
    format_now        = datetime.datetime.now().strftime("%Y%m%d_%H%M")    # '20260602_1435'
    frame_folder_name = f"frames_{video_name}_{format_now}"                # 'frames_IMG_0909_20260602_1435'
    frame_folder_path = f"{frames_dir}/{frame_folder_name}"
    logger.debug(f"result: {frame_folder_path}")
    return frame_folder_path

def ensure_unique_constraint(cursor: Any) -> None:
    """idempotency. Adds (video_id, frame_number) as unique constraint to automatically ensure no duplicaton"""
    logger.debug("adding unique constraint...")
    try:
        cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
        logger.info("constraint added successfully")
    except Exception:
        logger.debug("unique constraint already exists — continuing")

def frames_already_extracted(cursor: Any, video_id: int) -> bool:
    """Guard 3 — returns True if frames for this video already exist in the DB."""
    logger.debug(f"checking if frames already extracted for video_id={video_id}")
    cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
    result = cursor.fetchone()[0] > 0
    logger.debug(f"frames already extracted: {result}")
    return result

def write_sidecar(frame_folder_path: str, video_path: str, sample_rate: int, start_seconds: float, end_seconds: float | None, frames_stored: int) -> None:
    """Write extraction metadata alongside the frames so store_annotations.py can read it."""
    sidecar = {
        'frame_source':     'regular',
        'video_path':       video_path,
        'sample_rate':      sample_rate,
        'start_seconds':    start_seconds,
        'end_seconds':      end_seconds,
        'frames_extracted': frames_stored,
        'extracted_at':     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(frame_folder_path, 'extraction_params.yaml')
    with open(path, 'w') as f:
        yaml.dump(sidecar, f, default_flow_style=False, sort_keys=False)

def extract_and_save_frames(video_path: str, frame_folder_path: str, sample_rate: int, start_seconds: float, end_seconds: float | None) -> int:
    """Opens video, extracts frames at sample_rate, saves as JPG. Returns number of frames saved."""
    cap = cv2.VideoCapture(video_path)
    fps         = round(cap.get(cv2.CAP_PROP_FPS))
    step        = max(1, round(fps / sample_rate))
    start_frame = int(start_seconds * fps)
    end_frame   = int(end_seconds * fps) if end_seconds else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"fps={fps}, sample_rate={sample_rate}, step={step}, frames={start_frame}→{end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    os.makedirs(frame_folder_path, exist_ok=True)

    frame_count   = start_frame
    frames_stored = 0

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_count - start_frame) % step == 0:
            frame_name = f"frame_{frame_count}_{os.path.splitext(os.path.basename(video_path))[0]}"
            filename   = f"{frame_folder_path}/{frame_name}.jpg"
            cv2.imwrite(filename, frame)
            frames_stored += 1
            logger.debug(f"saved frame {frame_count} → {filename}")

        frame_count += 1

    cap.release()
    return frames_stored

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(conn: Any) -> None:
    logger.info("starting...")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)['extract_frames']
    video_path    = cfg['video_path']
    frames_dir    = cfg['frames_dir']
    sample_rate   = cfg.get('sample_rate', 1)
    start_seconds = cfg.get('start_seconds', 0) or 0
    end_seconds   = cfg.get('end_seconds')

    frame_folder_path = build_path_storage_frames(video_path, frames_dir)

    cursor = conn.cursor()

    ensure_unique_constraint(cursor) # guard 1 - making sure no duplicates allowed in sql
    logger.debug(f"checking video exists: {video_path}")
    video_id = get_video_id(cursor, video_path) # guard 2 - check video exist

    logger.info("setup complete | %s", {"video_path": video_path, "frames_dir": frames_dir, "frame_folder": frame_folder_path, "database": conn.database, "video_id": video_id})

    if frames_already_extracted(cursor, video_id): # guard 3 - skip if already done
        logger.info(f"frames already exist for {video_path} — skipping extraction")
        return

    frames_stored = extract_and_save_frames(video_path, frame_folder_path, sample_rate, start_seconds, end_seconds)
    frames_registered = register_frames(conn, frame_folder_path, video_path)
    write_sidecar(frame_folder_path, video_path, sample_rate, start_seconds, end_seconds, frames_stored)
    logger.info(f"done — {frames_stored} frames saved to disk, {frames_registered} registered in MySQL")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main(get_connection())

