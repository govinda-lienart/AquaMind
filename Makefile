# AquaMind Pipeline
# Run any step with: make <target>
# Example: make track

# ── UTILS ─────────────────────────────────────────────────────────────────────
# Open the main config file in your editor
config:
	open config.yaml

# Dump MySQL aquamind database to mysql_backup/ with timestamp
backup-db:
	docker exec cont-aquamind-sql mysqldump -u root -paquamind aquamind > mysql_backup/aquamind_$$(date +%Y%m%d_%H%M).sql


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA INGESTION
# Register videos in MySQL, then extract frames at 1fps
# ══════════════════════════════════════════════════════════════════════════════

# Read video_metadata xlsx and register video metadata in MySQL videos table
sync-videos:
	python -m scripts.sync_videos

# Extract 1 frame per second from video → save JPGs → store paths in MySQL frames table
extract-frames:
	python -m scripts.extract_frames


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ANNOTATION (LabelStudio)
# Import frames, label in LabelStudio, export annotations
# ══════════════════════════════════════════════════════════════════════════════

# Back up all LabelStudio projects as JSON to a dated folder in labelstudio_backup/
backup-labelstudio:
	python -m scripts.backup_labelstudio

# Create a new LabelStudio project and upload frames for labeling
# Auto-detects latest crossing_frames folder if frames_dir is empty in config
upload-labelstudio:
	python -m scripts.upload_labelstudio

# Download labeled annotations from LabelStudio as YOLO .txt files
download-labelstudio:
	python -m scripts.download_labelstudio

# Parse exported YOLO .txt files → store bboxes + keypoints in MySQL annotations table
store-annotations:
	python -m scripts.store_annotations

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DATASET PREPARATION
# Build the YOLO dataset folder from MySQL annotations
# ══════════════════════════════════════════════════════════════════════════════

# Query MySQL for annotated frames → build YOLO dataset folder with 80/20 train/val split
# Supports multiple annotation sessions merged into one dataset
prepare-dataset:
	python -m scripts.prepare_dataset

# Version the dataset with DVC and push to remote storage, then commit to git
push-dataset:
	dvc add dataset
	dvc push
	git add dataset.yaml dataset.dvc
	git commit -m "update dataset: $$(date +%Y-%m-%d)" || true
	git push


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TRAINING (Kaggle)
# Fine-tune YOLOv8 on the prepared dataset using Kaggle GPU
# ══════════════════════════════════════════════════════════════════════════════

# Open the Kaggle training notebook in the browser (run manually on Kaggle)
train:
	open https://www.kaggle.com/code/govindalienart/aquamind-train-yolov8

# Check the status of the Kaggle training kernel (running / complete / failed)
kaggle-status:
	kaggle kernels status govindalienart/aquamind-train-yolov8

# Poll Kaggle status every 60 seconds until you manually stop it
watch-kaggle:
	while true; do kaggle kernels status govindalienart/aquamind-train-yolov8; sleep 60; done

download-kaggle:
	kaggle kernels output govindalienart/aquamind-train-yolov8 -p runs/

# Log trained model weights, metrics, and artifacts to MLflow for experiment tracking
log-mlflow:
	python -m scripts.log_artifact_mlflow

run-mlflow:
	mlflow ui --backend-store-uri mlruns/ --port 5001



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — TRACKING
# Run the custom Kalman tracker on video using the trained YOLO model
# ══════════════════════════════════════════════════════════════════════════════

# Run YOLO detection + Kalman tracker on input video → annotated output video
# Logs Overlap detected / Crossing started events to logs/ automatically
track:
	python -m scripts.track_zebrafish


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — HARD NEGATIVE MINING (Active Learning)
# Find frames where fish overlap, extract them, re-label, retrain
# ══════════════════════════════════════════════════════════════════════════════

# Parse tracker log for bbox IoU > 0.5 overlap events → extract one frame per event
# Deduplicates within 5-frame windows to avoid near-identical frames
# Shows frame count preview and waits for Enter before extracting
extract-crossings:
	python -m scripts.extract_crossing_frames

# Start the YOLO ML backend server on port 9090 for LabelStudio auto-labeling
# Keep this terminal open while using LabelStudio
# Connect via: LabelStudio → Settings → Machine Learning → http://host.docker.internal:9090
ml-backend:
	python -m scripts.ml_backend


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

test:
	pytest -v

# ══════════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "══════════════════════════════════════════════"
	@echo "  AquaMind Pipeline"
	@echo "══════════════════════════════════════════════"
	@echo ""
	@echo "  STAGE 1 — Data Ingestion"
	@echo "  make sync-videos          Register videos in MySQL"
	@echo "  make extract-frames       Extract 1fps frames → MySQL"
	@echo ""
	@echo "  STAGE 2 — Annotation"
	@echo "  make backup-labelstudio   Back up all LS projects as JSON"
	@echo "  make upload-labelstudio   Create LS project + upload frames"
	@echo "  make download-labelstudio Download annotations from LabelStudio"
	@echo "  make store-annotations    Store annotations in MySQL"
	@echo ""
	@echo "  STAGE 3 — Dataset"
	@echo "  make prepare-dataset      Build YOLO dataset from MySQL (80/20)"
	@echo "  make push-dataset         DVC add + push + git commit dataset.dvc"
	@echo ""
	@echo "  STAGE 4 — Training"
	@echo "  make train                Open Kaggle training notebook"
	@echo "  make kaggle-status        Check Kaggle kernel status"
	@echo "  make watch-kaggle         Poll Kaggle status every 60s"
	@echo "  make download-kaggle      Download Kaggle run output to runs/"
	@echo "  make log-mlflow           Log artifacts + metrics to MLflow"
	@echo "  make run-mlflow           Launch MLflow UI in browser"
	@echo ""
	@echo "  STAGE 5 — Tracking"
	@echo "  make track                Run Kalman tracker → annotated video + log"
	@echo ""
	@echo "  STAGE 6 — Hard Negative Mining"
	@echo "  make extract-crossings    Extract overlap frames from tracker log"
	@echo "  make ml-backend           Start YOLO server for LS auto-labeling"
	@echo ""
	@echo "  make test                 Run all tests"
	@echo "  make config               Open config.yaml"
	@echo "  make backup-db            Dump MySQL to mysql_backup/"
	@echo ""
