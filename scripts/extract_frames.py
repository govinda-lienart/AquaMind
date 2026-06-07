# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import datetime

import cv2
import yaml

from scripts.db import get_connection, get_video_id


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def ensure_unique_constraint(cursor):
    try:
        cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
    except Exception:
        print("Unique constraint already exists — continuing")


def frames_already_extracted(cursor, video_id):
    cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
    return cursor.fetchone()[0] > 0


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    video_path = cfg['extract_frames']['video_path']
    frames_dir = cfg['extract_frames']['frames_dir']

    video_name        = os.path.splitext(os.path.basename(video_path))[0]  # 'IMG_0909'
    format_now        = datetime.datetime.now().strftime("%Y%m%d_%H%M")    # '20260602_1435'
    frame_folder_name = f"frames_{video_name}_{format_now}"                # 'frames_IMG_0909_20260602_1435'
    frame_folder_path = f"{frames_dir}/{frame_folder_name}"

    print(f"Config loaded: {cfg['extract_frames']}")

    with get_connection() as conn:
        cursor = conn.cursor()
        print("SQL connection:", conn.is_connected())

        ensure_unique_constraint(cursor)

        video_id = get_video_id(cursor, video_path)
        print(f"video_id {video_id} found for {video_path}")

        if frames_already_extracted(cursor, video_id):
            print(f"Frames for {video_path} already exist. Delete them first to re-extract.")
            return

        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print(f"FPS: {fps}")

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(frame_folder_path, exist_ok=True)

        frame_count = 0

        while True:
            ret, frame = cap.read()  # ret = False when video ends
            if not ret:
                break

            if frame_count % fps == 0:  # 1 frame per second
                print(f"saving frame {frame_count}")
                frame_name = f"frame_{frame_count}_{video_name}_{format_now}"
                filename   = f"{frame_folder_path}/{frame_name}.png"
                cv2.imwrite(filename, frame)
                cursor.execute(
                    "INSERT INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (video_id, filename, frame_count, frame_count / fps, datetime.datetime.now())
                )

            frame_count += 1

        conn.commit()
        cap.release()

    print(f"Done. Frames saved to {frame_folder_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
