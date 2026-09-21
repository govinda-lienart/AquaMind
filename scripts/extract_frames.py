"""
Extracts/stores frames (typically - one frame per second) from a video and stores the frame paths in MySQL.

Usage: python -m scripts.extract_frames
"""

# ── STEP 0: IMPORTS ───────────────────────────────────────────────────────────
# (we add each import only when the step that needs it appears)

import os
import yaml
import datetime
import cv2

# module imports
from scripts.console import banner, banner_sub
from scripts.db import get_connection, get_video_id

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── STEP 1: READ config.yaml ──────────────────────────────────────────────────

banner("STEP 1 - READ config.yaml")

banner_sub("load extract_frames section of config.yaml")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["extract_frames"]
logger.info(cfg)

banner_sub("extract_frames specific parameters")
video_path = cfg["video_path"]
frames_dir = cfg["frames_dir"]
sample_rate = cfg.get("sample_rate", 1)              # missing -> 1 frame per second
start_seconds = cfg.get("start_seconds", 0) or 0     # missing or blank (None) -> 0
end_seconds = cfg.get("end_seconds")                 # missing or blank -> None (whole video)

logger.info(f"video_path={video_path}, frames_dir={frames_dir}, sample_rate={sample_rate}, start_seconds={start_seconds}, end_seconds={end_seconds}")

# ── STEP 2: BUILD the output folder path ──────────────────────────────────────

banner("STEP 2 - BUILD output folder path")

video_filename = os.path.basename(video_path)        # 'videos/IMG_1839.MOV' -> 'IMG_1839.MOV'
video_name = os.path.splitext(video_filename)[0]     # 'IMG_1839.MOV' -> ('IMG_1839', '.MOV') -> 'IMG_1839'
logger.info(f"video_name={video_name}")
format_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
logger.info(f"format_now={format_now}")
frame_folder_path = f"{frames_dir}/frames_{video_name}_{format_now}"   # -> 'frames/frames_IMG_1839_20260919_1035'
logger.info(f"frame_folder_path={frame_folder_path}")

# ── STEP 3: CONNECT to MySQL ──────────────────────────────────────────────────

banner("STEP 3 - CONNECT to MySQL")

conn = get_connection()
logger.info(f"connected to database={conn.database}")
cursor = conn.cursor()

# ── STEP 4: GUARDRAILS (run in order) ─────────────────────────────────────────

banner("STEP 4 - GUARDRAILS")

banner_sub("guard 1 - unique constraint on frames (video_id, frame_number)") # never allow the same frame of the same video to be stored twice
try:
    cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
    logger.info("contrain added")
except Exception:
    logger.info("constrain already exist, continuing") # eats the error in case constrains already established

banner_sub("guard 2 - video is registered in the videos table") 
video_id = get_video_id(cursor, video_path) # has 2 functions: Check the video is registered. If it isn't, it stops with an error. 2) Return its id. That number is the foreign key you attach to every frame row later.
logger.info(f"video_id={video_id}, type={type(video_id)}")
cursor.execute("SELECT fps FROM videos WHERE id = %s", (video_id,))   # the old register_frames() took timestamps from videos.fps, not from OpenCV
video_fps = cursor.fetchone()[0]
logger.info(f"video_fps (from videos table)={video_fps}")

banner_sub("guard 3 - frames already extracted for this video?")
cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))  # values go in a tuple; one value needs a trailing comma
already_extracted = cursor.fetchone()[0] > 0
logger.info(f"already_extracted={already_extracted}")
if already_extracted:
    logger.info(f"frames already exist for {video_path}, skipping extraction")
    raise SystemExit

# ── STEP 5: EXTRACT frames to disk ────────────────────────────────────────────

banner("STEP 5 - EXTRACT frames to disk")

banner_sub("video settings: fps / step / start-end frame")

cap = cv2.VideoCapture(video_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
logger.info(f"total_frames = {total_frames}")

step = max(1, round(fps / sample_rate)) # : 60 / 1 = 60 (keep every 60th frame), 60 / 4 = 15 (keep every 15th).
start_frame = int(start_seconds * fps) # so 50 sec x 60 fr/sec is frame 3000
end_frame = int(end_seconds * fps) if end_seconds else total_frames  # end time given -> convert to frame; None -> use last frame of video
logger.info(f"fps={fps}, sample_rate={sample_rate}, step={step}")
logger.info(f"frames {start_frame} -> {end_frame} (total {total_frames})")

banner_sub("saving frames")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
os.makedirs(frame_folder_path, exist_ok=True) # putting it here and not step 2 makes sure the folder is not created in case the guardrails detect an error.

frame_count = start_frame # counter
frames_stored = 0
rows = []   # one tuple per saved frame, inserted in STEP 6 [(1, "frames/f_3000.jpg", 3000, 50.0),    (1, "frames/f_3060.jpg", 3060, 51.0),


while frame_count < end_frame:
    if (frame_count - start_frame) % step == 0:   # frames walked since the start is a multiple of step -> keep it, otherwise skip (start 100, step 15: keep 100, 115, 130...)
        ret, frame = cap.read()   # read the frame (ret = False at end of video) #Every call to cap.read() moves the pointer forward by one frame.
        if not ret:  
            break # break when The video ran out of frames or error
        filename = f"{frame_folder_path}/frame_{frame_count}_{video_name}.jpg"
        cv2.imwrite(filename, frame) # save it as a JPG
        rows.append((video_id, filename, frame_count, frame_count / video_fps, datetime.datetime.now()))   # order must match the INSERT in STEP 6 - important step as this one will be used for storying in mysql
        frames_stored += 1
    else: # if == 0 is false  - skip the frame with grab().
        if not cap.grab(): # skip a frame without decoding it (fast); still moves the bookmark forward
            break  # break if cap.grab() did not succeed
    frame_count += 1 # every round moves one frame forward, kept or skipped
cap.release()
logger.info(f"frames_stored={frames_stored}")

# ── STEP 6: REGISTER frames in MySQL ──────────────────────────────────────────

banner("STEP 6 - REGISTER frames in MySQL")

cursor.executemany( # allows to loop over the rows
    "INSERT IGNORE INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) VALUES (%s, %s, %s, %s, %s)", # IGNORE = duplicates skipped, safe to re-run (as in the old register_frames); usually one row with several columns but here several rows several coluimsn
    rows, # see list of tuples in step 5
)
conn.commit()   # nothing is saved in MySQL until this line
logger.info(f"rows inserted = {cursor.rowcount} (frames_stored = {frames_stored})")

# ── STEP 7: SIDECAR + DONE ────────────────────────────────────────────────────
# lineage: extraction_params.yaml -> store_annotations.py -> annotation_sets -> dataset_card.yaml -> MLflow
banner("STEP 7 - SIDECAR + DONE")

banner_sub("write extraction_params.yaml")

params = {
    "frame_source": "regular",              # read by store_annotations.py
    "video_path": video_path,
    "sample_rate": sample_rate,             # read by store_annotations.py
    "start_seconds": start_seconds,         # read by store_annotations.py
    "end_seconds": end_seconds,             # read by store_annotations.py
    "frames_extracted": frames_stored,      # read by store_annotations.py (key name must stay 'frames_extracted')
    "extracted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "fps": fps,                             # extras, not read by store_annotations.py
    "step": step,
    "start_frame": start_frame,
    "end_frame": end_frame,
}

with open(f"{frame_folder_path}/extraction_params.yaml", "w") as f:   # "w" = write (STEP 1 used the default "r" = read)
    yaml.safe_dump(params, f, sort_keys=False) # dump = write a dict to YAML (load = read); sort_keys=False keeps my order

banner_sub("summary")
logger.info(f"done: {frames_stored} frames saved to {frame_folder_path} and {cursor.rowcount} rows registered in MySQL")
cursor.close()
conn.close()

