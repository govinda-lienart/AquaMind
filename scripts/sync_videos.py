"""Registers each video from video_metadata.xlsx into the MySQL videos table.
   Usage: python scripts/sync_videos.py
"""

# IMPORTS
import logging
import openpyxl
import cv2
from scripts.console import banner, banner_sub
from scripts.db import get_connection

# CONSTANTS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# HELPER FUNCTIONS

def load_video_row(xlsx_path): # returns a list of dicts, one per video row
    """ loads an excel file and transform it into a dictionary """
    wb = openpyxl.load_workbook(xlsx_path) # creates a workbook object
    ws = wb["videos"] # grab the sheet videos
    headers = [cell.value for cell in ws[1]] # values
    rows = ws.iter_rows(min_row=2, values_only=True) # row min skip row 1 with headers # generator
    return [dict(zip(headers, row)) for row in rows] # returns a dictionary same as the loop version: zip each row against headers, wrap in dict, collect


def get_video_fps_resolution(file_path):
    """extract key video data from the video like FPS, width"""
    cap = cv2.VideoCapture(file_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS)) # FPS = 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1920
    if width >= 3840:
        resolution = '4K'
    elif width >= 1920:
        resolution = '1080p'
    else:
        resolution = '720p'
    cap.release()
    return fps, resolution


def register_video(row, conn):
    fps, resolution = get_video_fps_resolution(row["file_path"])  # reading that video path to extract automaitcally fps and resolution
    cursor = conn.cursor()
    cursor.execute( # ON DUPLICATE KEY UPDATE: safe to re-run — updates existing row instead of erroring on duplicate file_path
        """INSERT INTO videos (file_path, fps, resolution, activity, plants, fish_count, notes,
                                filmed_at, species, morph, tank_width_cm, tank_height_cm, tank_depth_cm)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            fps            = VALUES(fps),
            resolution     = VALUES(resolution),
            activity       = VALUES(activity),
            plants         = VALUES(plants),
            fish_count     = VALUES(fish_count),
            notes          = VALUES(notes),
            filmed_at      = VALUES(filmed_at),
            species        = VALUES(species),
            morph          = VALUES(morph),
            tank_width_cm  = VALUES(tank_width_cm),
            tank_height_cm = VALUES(tank_height_cm),
            tank_depth_cm  = VALUES(tank_depth_cm)""",
        (row["file_path"], fps, resolution, row["activity"], row["plants"], row["fish_count"],
        row["notes"], row["filmed_at"], row["species"], row["morph"],
        row["tank_width_cm"], row["tank_height_cm"], row["tank_depth_cm"])
    )
    conn.commit()
    cursor.close()
    logger.info(f"synced {row['file_path']} fps={fps} resolution={resolution}")

# MAIN

banner("SYNC VIDEOS")

banner_sub("Loading video metadata from xlsx")
video_rows = load_video_row("video_metadata.xlsx")

banner_sub("Connecting to MySQL")
conn = get_connection()

banner_sub("Syncing all videos to MySQL")
for row in video_rows:
    register_video(row, conn)

conn.close()