"""
Shared database utilities for the AquaMind pipeline.

Provides get_connection() and reusable query helpers imported by all scripts.
Credentials are read from .env (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME).
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os

import cv2
import mysql.connector
from dotenv import load_dotenv


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


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


def register_video(file_path, session_type, obstacles, fish_count, notes=None,
                   species='danio_rerio', morph=None,
                   tank_width_cm=None, tank_height_cm=None, tank_depth_cm=None,
                   filmed_at=None):
    """Registers a video in MySQL — reads FPS and resolution directly from the video file."""
    cap        = cv2.VideoCapture(file_path)
    fps        = int(cap.get(cv2.CAP_PROP_FPS))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resolution = '4K' if width >= 3840 else '1080p' if width >= 1920 else '720p'
    cap.release()
    logger.debug(f"registering {file_path} fps={fps} resolution={resolution}")

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT IGNORE INTO videos (file_path, fps, resolution, session_type, obstacles, fish_count, notes, filmed_at,
                                   species, morph, tank_width_cm, tank_height_cm, tank_depth_cm)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (file_path, fps, resolution, session_type, obstacles, fish_count, notes, filmed_at,
          species, morph, tank_width_cm, tank_height_cm, tank_depth_cm))
    conn.commit()
    video_id = cursor.lastrowid
    cursor.close()
    conn.close()
    logger.debug(f"register_video result: video_id={video_id}")
    return video_id


def get_video_id(cursor, video_path):
    cursor.execute("SELECT id FROM videos WHERE file_path = %s", (video_path,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Video {video_path} not registered. Run sync_videos.py first.")
    logger.debug(f"get_video_id: {video_path} → id={row[0]}")
    return row[0]


def get_frame_id(cursor, frames_folder, frame_number):
    pattern = f"{frames_folder}/frame_{frame_number}%.png"
    cursor.execute("SELECT id FROM frames WHERE frame_path LIKE %s", (pattern,))
    row = cursor.fetchone()
    cursor.fetchall()  # drain cursor before reusing
    if row is None:
        raise ValueError(f"No DB record found for frame_{frame_number} in {frames_folder}")
    logger.debug(f"get_frame_id: frame_{frame_number} → id={row[0]}")
    return row[0]
