"""
Extracts/stores one frame per second from a video and stores the frame paths in MySQL.

Input  : video file registered in the videos table (run sync_videos.py first)
Output : JPG files on disk + rows inserted into the frames table
Guards : unique constraint (no duplicates), video must be registered, skips if already extracted
"""

# ── STEP 0: IMPORTS ───────────────────────────────────────────────────────────
# (we add each import only when the step that needs it appears)


# ── STEP 1: READ config.yaml ──────────────────────────────────────────────────
# TODO: open config.yaml, load it, keep only the ['extract_frames'] section -> cfg
# TODO: pull out video_path, frames_dir, sample_rate, start_seconds, end_seconds
# TODO: print them to check they are what you expect


# ── STEP 2: BUILD the output folder path ──────────────────────────────────────
# TODO: video_name from video_path, timestamp from now -> frame_folder_path
# TODO: print it


# ── STEP 3: CONNECT to MySQL ──────────────────────────────────────────────────
# TODO: conn = get_connection()  (from scripts.db), then cursor = conn.cursor()


# ── STEP 4: GUARDRAILS (run in order) ─────────────────────────────────────────
# TODO guard 1: unique constraint on frames (video_id, frame_number)
# TODO guard 2: get_video_id(cursor, video_path)
# TODO guard 3: frames already extracted? -> stop if yes


# ── STEP 5: EXTRACT frames to disk ────────────────────────────────────────────
# TODO: open video with cv2, work out fps / step / start_frame / end_frame
# TODO: loop, save every Nth frame as JPG into frame_folder_path
# TODO: count frames_stored


# ── STEP 6: REGISTER frames in MySQL ──────────────────────────────────────────
# TODO: scan frame_folder_path, INSERT IGNORE one row per JPG into frames


# ── STEP 7: SIDECAR + DONE ────────────────────────────────────────────────────
# TODO: write extraction_params.yaml into frame_folder_path
# TODO: log the summary
