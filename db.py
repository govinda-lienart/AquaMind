import mysql.connector
import cv2
import os
from datetime import datetime
from dotenv import load_dotenv


# connect with sql

def get_connection():
    load_dotenv()
    return mysql.connector.connect(
        host     = os.getenv("DB_HOST"),
        port     = int(os.getenv("DB_PORT")),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        database = os.getenv("DB_NAME")
    )

# REGISTER A VIDEO IN SQL

def register_video(file_path, session_type, obstacles, fish_count, notes=None,
                   species='danio_rerio', morph=None,
                   tank_width_cm=None, tank_height_cm=None, tank_depth_cm=None):
    cap = cv2.VideoCapture(file_path)
    fps         = int(cap.get(cv2.CAP_PROP_FPS))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resolution  = '4K' if width >= 3840 else '1080p' if width >= 1920 else '720p'
    cap.release()

    filmed_at = datetime.fromtimestamp(os.path.getctime(file_path))

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT IGNORE INTO videos (file_path, fps, resolution, session_type, obstacles, fish_count, notes, filmed_at,
                                   species, morph, tank_width_cm, tank_height_cm, tank_depth_cm)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (file_path, fps, resolution, session_type, obstacles, fish_count, notes, filmed_at,
          species, morph, tank_width_cm, tank_height_cm, tank_depth_cm)) # ignore  works because I used ALTER TABLE videos ADD UNIQUE (file_path)
    conn.commit()
    video_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return video_id

