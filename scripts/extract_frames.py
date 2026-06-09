"""
Extracts/stores one frame per second from a video and stores the frame paths in MySQL.

Input  : video file registered in the videos table (run sync_videos.py first)
Output : PNG files on disk + rows inserted into the frames table
Guards : unique constraint (no duplicates), video must be registered, skips if already extracted
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import datetime
import logging
import re
import yaml
import os
import cv2
from scripts.db import get_connection, aquatest_connection, get_video_id

# ── LOGGER ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'

# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_path_storage_frames(video_path, frames_dir):
    """ Bulding path storage from loadad data config file"""
    logger.debug(f"video_path={video_path}, frames_dir={frames_dir}")
    video_name        = os.path.splitext(os.path.basename(video_path))[0]  # 'IMG_0909'
    format_now        = datetime.datetime.now().strftime("%Y%m%d_%H%M")    # '20260602_1435'
    frame_folder_name = f"frames_{video_name}_{format_now}"                # 'frames_IMG_0909_20260602_1435'
    frame_folder_path = f"{frames_dir}/{frame_folder_name}"
    logger.debug(f"result: {frame_folder_path}")
    return frame_folder_path

def ensure_unique_constraint(cursor):
    """idempotency. Adds (video_id, frame_number) as unique constraint to automatically ensure no duplicaton"""
    logger.debug("adding unique constraint...")
    try:
        cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
        logger.info("constraint added successfully")
    except Exception:
        logger.debug("unique constraint already exists — continuing")

def frames_already_extracted(cursor, video_id):
    """Guard 3 — returns True if frames for this video already exist in the DB."""
    logger.debug(f"checking if frames already extracted for video_id={video_id}")
    cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
    result = cursor.fetchone()[0] > 0
    logger.debug(f"frames already extracted: {result}")
    return result

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(conn, video_path=None, frames_dir=None):
    logger.info("starting...")
    if video_path is None: # if video_path explicity provided in argument - test mode
        cfg = yaml.safe_load(open(CONFIG_PATH))
        video_path = cfg['extract_frames']['video_path']
        frames_dir = cfg['extract_frames']['frames_dir']

    frame_folder_path = build_path_storage_frames(video_path, frames_dir)

    cursor = conn.cursor()

    ensure_unique_constraint(cursor) # guard 1 - makeing sure no duplicates allowed in sql
    logger.debug(f"checking video exists: {video_path}")
    video_id = get_video_id(cursor, video_path) # guard 2 - check video exist

    logger.info("setup complete | %s", {"video_path": video_path, "frames_dir": frames_dir, "frame_folder": frame_folder_path, "database": conn.database, "video_id": video_id})

    if frames_already_extracted(cursor, video_id): # guard 3 - skip if already done
        logger.info(f"frames already exist for {video_path} — skipping extraction")
        return

    # ── EXTRACTION LOOP ───────────────────────────────────────────────────────

    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"fps: {fps}")

    os.makedirs(frame_folder_path, exist_ok=True)

    frame_count   = 0
    frames_stored = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            frame_name = f"frame_{frame_count}_{os.path.splitext(os.path.basename(video_path))[0]}"
            filename   = f"{frame_folder_path}/{frame_name}.png"
            cv2.imwrite(filename, frame)
            cursor.execute(
                "INSERT INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (video_id, filename, frame_count, frame_count / fps, datetime.datetime.now())
            )
            frames_stored += 1
            logger.debug(f"saved frame {frame_count} → {filename}")

        frame_count += 1

    conn.commit()
    cap.release()
    logger.info(f"done — {frames_stored} frames saved to {frame_folder_path}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main(get_connection())

# ── TESTS ─────────────────────────────────────────────────────────────────────
#  pytest scripts/extract_frames.py -v -s

def test_main():
    print ("\n*****************************************************")
    print("\n--- testing: main ---")
    conn = aquatest_connection()
    main(conn, video_path='videos/IMG_0350.MOV', frames_dir='frames')
    conn.close()

def test_ensure_unique_constraint():
    print ("\n*****************************************************")
    print("\n--- testing: ensure_unique_constraint ---")
    conn = aquatest_connection()
    cursor = conn.cursor()
    ensure_unique_constraint(cursor)
    ensure_unique_constraint(cursor)
    conn.close()

def test_frames_already_extracted():
    print("\n*****************************************************")
    print("\n--- testing: frames_already_extracted ---")
    conn = aquatest_connection()
    try:
        cursor = conn.cursor()
        assert frames_already_extracted(cursor, 9999) == False  # unknown video_id → no frames
        assert frames_already_extracted(cursor, 19)   == True   # fixtures.sql seeds frames for video_id=19
    finally:
        conn.close()

def test_build_path_storage_frames():
    print ("\n*****************************************************")
    print("--- testing: build_path_storage_frames ---\n")
    result = build_path_storage_frames('videos/IMG_0350.MOV', 'frames')
    assert re.match(r'frames/frames_IMG_0350_\d{8}_\d{4}$', result), f"Unexpected path format: {result}"
