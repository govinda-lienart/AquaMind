# IMPORTS
import cv2
import os
from datetime import datetime
from sympy.simplify.hyperexpand import try_lerchphi
import yaml

from extract_frames import format_now, video_path

# LOAD CONFIG

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# BUILD FRAME FOLDER NAME eg. frames_IMG_0909_20260602_1435'

datetime_now = datetime.now().strftime("%Y%M%d_%H%M")
video_path = ['extract_frame']['video_path']  # videos/IMG_0909.MOV
video_name_ext = os.path.basename(video_path)
video_name = os.path.splitext(video_name_ext)[0]
video_folder_name = f'frames_{video_name}_{datetime_now}'

# CONNECTING WITH MYSQL — context manager closes connection automatically at end of block

from db import get_connection

with get_connection() as conn:
    cursor = conn.cursor

    # make it unique
    try:
        cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
    except:
        print("all good")

    
    
    cursor.execute("SELECT video_id FROM videos WHERE file_path = %s",(video_path))
    row = cursor.fetchone()
    if not row:
        print("video not registered")








    # extract_frames:
    # video_path: videos/IMG_0909.MOV
    # frames_dir: frames





# CONNECT TO MYSQL
    # IDEMPOTENCY
    # CHECK VIDEO REGISTERED
    # CHECK ALREADY EXTRACTED
    # LOAD VIDEO
    # CREATE DIRECTORIES
    # GET FPS
    # EXTRACT LOOP
    # COMMIT
    # CLOSE VIDEO