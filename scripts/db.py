"""
Shared database utilities for the AquaMind pipeline.

Provides get_connection() and reusable query helpers imported by all scripts.
Credentials are read from .env (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME).
"""

from __future__ import annotations # avoid type hint issues when import


# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
from typing import Any

import cv2
import mysql.connector # 
from dotenv import load_dotenv


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__) # log


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def get_connection():
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


def get_video_id(cursor: Any, video_path: str) -> int:
    cursor.execute("SELECT id FROM videos WHERE file_path LIKE %s", (f"%{video_path}%",))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Video '{video_path}' not found in videos table — run sync_videos.py first.")
    logger.debug(f"get_video_id: {video_path} → id={row[0]}")
    return row[0]


def get_frame_id(cursor: Any, frames_folder: str, frame_number: int) -> int:
    for pattern in [
        f"{frames_folder}/frame_{frame_number}_%",       # 1fps frames: frame_60_IMG_0350.png
        f"{frames_folder}/frame_{frame_number:06d}.%",   # crossing frames: frame_002080.jpg
    ]:
        cursor.execute("SELECT id FROM frames WHERE frame_path LIKE %s", (pattern,))
        row = cursor.fetchone()
        cursor.fetchall()
        if row is not None:
            logger.debug(f"get_frame_id: frame_{frame_number} → id={row[0]}")
            return row[0]
    raise ValueError(f"No DB record found for frame {frame_number} in {frames_folder}")


def register_frames(conn: Any, frames_folder: str, video_path: str) -> int:
    """Register all frame_*.jpg/png files in a folder into MySQL. Safe to re-run (INSERT IGNORE)."""
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

        