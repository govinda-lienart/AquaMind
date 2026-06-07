# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os

import cv2
import mysql.connector
from dotenv import load_dotenv


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def get_connection():
    load_dotenv()
    return mysql.connector.connect(
        host     = os.getenv("DB_HOST"),
        port     = int(os.getenv("DB_PORT")),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        database = os.getenv("DB_NAME")
    )


def register_video(file_path, session_type, obstacles, fish_count, notes=None,
                   species='danio_rerio', morph=None,
                   tank_width_cm=None, tank_height_cm=None, tank_depth_cm=None,
                   filmed_at=None):
    cap        = cv2.VideoCapture(file_path)
    fps        = int(cap.get(cv2.CAP_PROP_FPS))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resolution = '4K' if width >= 3840 else '1080p' if width >= 1920 else '720p'
    cap.release()

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
    return video_id


def get_video_id(cursor, video_path):
    cursor.execute("SELECT id FROM videos WHERE file_path = %s", (video_path,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Video {video_path} not registered. Run sync_videos.py first.")
    return row[0]


def get_frame_id(cursor, frames_folder, frame_number):
    pattern = f"{frames_folder}/frame_{frame_number}%.png"
    cursor.execute("SELECT id FROM frames WHERE frame_path LIKE %s", (pattern,))
    row = cursor.fetchone()
    cursor.fetchall()  # drain cursor before reusing
    if row is None:
        raise ValueError(f"No DB record found for frame_{frame_number} in {frames_folder}")
    return row[0]
