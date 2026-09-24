"""
Shared database utilities for the AquaMind pipeline.

Provides get_connection() and reusable query helpers imported by all scripts.
Credentials are read from .env (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME).
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os

import cv2
import mysql.connector # 
from dotenv import load_dotenv


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__) # log


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def get_connection():
    """Opens a MySQL connection using the credentials in .env and returns it.
    The caller creates cursors from it and must commit() to save any INSERTs."""
    load_dotenv()
    db_name = os.getenv("DB_NAME")
    logger.debug(f"connecting to database={db_name}")
    return mysql.connector.connect(
        host     = os.getenv("DB_HOST"),
        port     = int(os.getenv("DB_PORT")),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        database = db_name
    )


def get_video_id(cursor, video_path):
    """Returns videos.id for the video whose file_path contains video_path (e.g. "IMG_0867").
    Raises ValueError if the video isn't registered yet (run sync_videos.py first)."""
    cursor.execute("SELECT id FROM videos WHERE file_path LIKE %s", (f"%{video_path}%",))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Video '{video_path}' not found in videos table — run sync_videos.py first.")
    logger.debug(f"get_video_id: {video_path} → id={row[0]}")
    return row[0]


def get_frame_id(cursor, frames_folder, frame_number): # get_frame_id(reading_cursor, "frames/frames_IMG_0867_20260921_1357", 0)
    """Returns frames.id for a frame number inside a frames folder.
    Tries both filename styles (1fps frame_60_IMG_0350.png, crossing frame_002080.jpg).
    Raises ValueError if no matching row is registered in the frames table."""
    for pattern in [
        f"{frames_folder}/frame_{frame_number}_%",       # 1fps frames: frames/frames_IMG_0867_20260921_1357/frame_0_% => The % is a wildcard, so it matches the stored path frames/frames_IMG_0867_20260921_1357/frame_0_IMG_0867.jpg. The query runs
        f"{frames_folder}/frame_{frame_number:06d}.%",   # crossing frames: frame_002080.jpg
    ]:
        cursor.execute("SELECT id FROM frames WHERE frame_path LIKE %s", (pattern,))
        row = cursor.fetchone() #  gives one row, a tuple with one number: the id of that frame's row in frames. => row = (2763,)   | 
        cursor.fetchall()  # fetchall() reads the leftover rows and throws them away, 
        if row is not None:
            logger.debug(f"get_frame_id: frame_{frame_number} → id={row[0]}")
            return row[0] #  row[0] = 2763
    raise ValueError(f"No DB record found for frame {frame_number} in {frames_folder}")


def register_frames(conn, frames_folder, video_path):
    """Registers every frame_*.jpg/png file in a folder as a row in the frames table.
    Looks up the video's id and fps first (fps gives each frame its timestamp), commits,
    and returns the number of NEW rows. Safe to re-run: INSERT IGNORE skips frames already registered."""
    import re
    from datetime import datetime

    #  look up the video 

    cursor = conn.cursor()
    cursor.execute("SELECT id, fps FROM videos WHERE file_path = %s", (video_path,)) # queries videos by file_path to get video_id and fps
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Video {video_path} not registered. Run sync_videos.py first.")
    video_id, fps = row

    # scan the folder + insert each frame — for each matching file, inserts a row into frames with:

    name_pattern = re.compile(r'frame_(\d+)[^.]*\.(jpg|png)$')
    registered = 0
    for fname in sorted(os.listdir(frames_folder)):
        m = name_pattern.match(fname)
        if not m:
            continue
        frame_number = int(m.group(1))
        frame_path   = f"{frames_folder}/{fname}"
        cursor.execute(
            "INSERT IGNORE INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (video_id, frame_path, frame_number, frame_number / fps, datetime.now())
        )
        registered += cursor.rowcount

    conn.commit()
    cursor.close()
    logger.debug(f"register_frames: {registered} new frames registered in {frames_folder}")
    return registered

        