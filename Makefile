# AquaMind Pipeline
# Run any step with: make <target>
# Example: make store-annotations

# ── STEP 1 — register videos in MySQL ─────────────────────────────────────────
sync-videos:
	python -m scripts.sync_videos

# ── STEP 2 — extract 1 frame/sec from video → MySQL frames table ──────────────
extract-frames:
	python -m scripts.extract_frames

# ── STEP 3 — export labeled annotations from Label Studio as YOLO .txt files ──
export-labelstudio:
	python -m scripts.export_labelstudio

# ── STEP 4 — parse YOLO .txt files → MySQL annotations + keypoints tables ─────
store-annotations:
	python -m scripts.store_annotations

# ── STEP 5 — build YOLO dataset folder from MySQL (80/20 split) ───────────────
prepare-dataset:
	python -m scripts.prepare_dataset

# ── STEP 6 — log YOLO training artifacts and metrics to MLflow ────────────────
log-mlflow:
	python -m scripts.log_artifact_mlflow

# ── STEP 7 — run custom Kalman tracker on video → annotated output video ──────
track:
	python -m scripts.track_zebrafish

# ── UTILS ─────────────────────────────────────────────────────────────────────
test:
	pytest -v

help:
	@echo ""
	@echo "AquaMind Pipeline Steps:"
	@echo "  make sync-videos        Step 1 — register videos in MySQL"
	@echo "  make extract-frames     Step 2 — extract frames from video"
	@echo "  make export-labelstudio Step 3 — export annotations from Label Studio"
	@echo "  make store-annotations  Step 4 — store annotations in MySQL"
	@echo "  make prepare-dataset    Step 5 — build YOLO dataset from MySQL"
	@echo "  make log-mlflow         Step 6 — log training run to MLflow"
	@echo "  make track              Step 7 — run Kalman tracker on video"
	@echo "  make test               Run all tests"
	@echo ""
