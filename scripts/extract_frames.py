# IMPORTS

import os
import cv2
from datetime import datetime
import yaml

# CHECK ACCESS CONFIGURATION FILE

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
print(f"Metadata access confirmed: {cfg['extract_frames']}")

# BUILDING NAME FOR FRAME FOLDER : frames/frames_IMG_0909_20260602_1435/

format_now = datetime.now().strftime("%Y%m%d_%H%M")            # '20260602_1435' 
video_path = cfg['extract_frames']['video_path']                # 'videos/IMG_0909.MOV'
video_name_ext = os.path.basename(video_path)                   # 'IMG_0909.MOV'
video_name = os.path.splitext(video_name_ext)[0]                # 'IMG_0909'
frame_folder_name = f"frames_{video_name}_{format_now}"         # 'frames_IMG_0909_20260602_1435'

# CONNECTING WITH MYSQL — context manager closes connection automatically at end of block
from scripts.db import get_connection
with get_connection() as conn:
    cursor = conn.cursor()
    print("SQL Connection:", conn.is_connected())

    # IDEMPOTENCY — pair (video_id, frame_number) must be unique across all rows
    # running this script 1 or 10 times produces the same result — no duplicates
    try:
        cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
    except:
        print("Unique constraint already established - continue")

    # CHECK IF VIDEO REGISTERED IN SQL
    cursor.execute("SELECT id FROM videos WHERE file_path = %s", (video_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Video {video_path} not registered. Run sync_videos.py first.")
        exit()
    video_id = row[0]
    print(f"video_id {video_id} found for {video_path}")

    # CHECK IF VIDEO ALREADY EXTRACTED
    cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
    if cursor.fetchone()[0] > 0:  # fetchone() returns tuple e.g. (548,) — [0] unpacks it
        print(f"Frames for {video_path} already exist. Delete them first to re-extract.")
        exit()

    # LOAD VIDEO
    cap = cv2.VideoCapture(video_path)

    # CREATE DIRECTORY FOR EXTRACTED FRAMES
    os.makedirs(cfg['extract_frames']["frames_dir"], exist_ok=True) # created directory frames/
    os.makedirs(f"{cfg['extract_frames']['frames_dir']}/{frame_folder_name}", exist_ok=True)

    # EXTRACT FRAME PER SECOND
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"Number of frames per second: {fps}")    

    # LOOP TO EXTRACT FRAME AND STORE DATA IN SQL DB
    frame_count = 0

    while True:
        ret, frame = cap.read()  # ret = True if frame available, False if video ended
                                 # frame = NumPy array (grid of pixels)
        if not ret:
            break

        if frame_count % fps == 0:  # select 1 frame per second (every 60th frame at 60fps)
            print(f"saving frame {frame_count}")
            frame_name = f"frame_{frame_count}_{video_name}_{format_now}"
            filename   = f"{cfg['extract_frames']['frames_dir']}/{frame_folder_name}/{frame_name}.png"
            cv2.imwrite(filename, frame)
            timestamp = frame_count / fps
            cursor.execute(
                "INSERT INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) VALUES (%s, %s, %s, %s, %s)",
                (video_id, filename, frame_count, timestamp, datetime.now())
            )
        frame_count += 1

    # SAVE TO SQL DB — commits all inserts permanently
    conn.commit()

    # CLOSE VIDEO
    cap.release()

    print(f"Done. Frames saved to {cfg['extract_frames']['frames_dir']}/{frame_folder_name}")




