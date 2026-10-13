"""
tracker.py - the fish tracker, Stage 4 core version.
Output: a per-run folder holding the annotated video, log, config sidecar and tracks.parquet

Run:  python -m scripts.tracker
"""

# imports


# ── STEP 1: load the config ────────────────────────────────────────────────────
# read the `tracker:` section of config.yaml into c, then pull out the settings we need


# ── STEP 2: open the video ─────────────────────────────────────────────────────
# open the video with cv2, read fps / width / height


# ── STEP 3: load the YOLO model ────────────────────────────────────────────────
# load_yolo() from scripts.model_registry understands models:/aquamind-yolo-detector@baseline


# ── STEP 4: fish memory ────────────────────────────────────────────────────────
# one dict per fish: id, x, y, vx, vy, bbox, confidence, hits, missing


# ── STEP 5: the frame loop ─────────────────────────────────────────────────────
# while True: read a frame, stop at the end of the clip

    # STEP 5a: detect, run YOLO on the frame, keep class 0 (danio_rerio), skip class 1 (reflection)

    # STEP 5b: predict each fish's next position + match predictions to boxes (Hungarian)

    # STEP 5c: ghosts, healthy fish match first, missing fish search a wider (capped) radius

    # STEP 5d: calibration lock, after calibration_secs keep the num_fish best tracks, give IDs 1..N

    # STEP 5e: logging, occlusion_lost / occlusion_recovery + crossings (two fish's boxes overlapping)

    # STEP 5f: store this frame's rows for tracks.parquet + draw boxes on the frame


# ── STEP 6: save outputs ───────────────────────────────────────────────────────
# tracks.parquet + annotated video
