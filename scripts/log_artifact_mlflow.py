"""
Logs a YOLO training run (metrics, artifacts, video metadata) to MLflow.

Input  : YOLO run folder path from config.yaml (contains results.csv + weights)
Needs  : dataset folder present, MySQL running for video metadata
Output : MLflow experiment run with params, per-epoch metrics, and artifacts
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os

import mlflow
import pandas as pd
import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'
MLRUNS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlruns')
EXPERIMENT  = 'aquamind'
YOLO_MODEL  = 'yolov8n'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])


def load_results_csv(run_path):
    """Loads YOLO results.csv — strips whitespace from column names (YOLO adds leading spaces)."""
    df = pd.read_csv(os.path.join(run_path, 'results.csv'))
    df.columns = df.columns.str.strip()
    return df


def log_epoch_metrics(df):
    for _, row in df.iterrows():
        epoch = int(row['epoch'])
        metrics = {
            'train/box_loss':  row['train/box_loss'],
            'train/cls_loss':  row['train/cls_loss'],
            'train/dfl_loss':  row['train/dfl_loss'],
            'val/box_loss':    row['val/box_loss'],
            'val/cls_loss':    row['val/cls_loss'],
            'val/dfl_loss':    row['val/dfl_loss'],
            'precision':       row['metrics/precision(B)'],
            'recall':          row['metrics/recall(B)'],
            'mAP50':           row['metrics/mAP50(B)'],
            'mAP50_95':        row['metrics/mAP50-95(B)'],
            'lr0':             row['lr/pg0'],
            'lr1':             row['lr/pg1'],
            'lr2':             row['lr/pg2'],
        }
        mlflow.log_metrics(metrics, step=epoch)
    logger.info(f"{len(df)} epochs logged")


def log_dataset_card(dataset_path):
    card_path = os.path.join(dataset_path, 'dataset_card.yaml')
    if os.path.exists(card_path):
        mlflow.log_artifact(card_path)
        logger.info(f"dataset card logged from {card_path}")
    else:
        logger.warning(f"no dataset_card.yaml found in {dataset_path} — skipping")


def fetch_video_sources(conn, annotation_set_ids):
    """Fetches video metadata for all videos that contributed to the annotation sets."""
    placeholders = ','.join(['%s'] * len(annotation_set_ids))
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        f"""SELECT DISTINCT v.file_path, v.filmed_at, v.activity, v.plants, v.fish_count, v.notes
            FROM videos v
            JOIN frames fr ON fr.video_id = v.id
            JOIN annotations a ON a.frame_id = fr.id
            WHERE a.annotation_set_id IN ({placeholders})""",
        annotation_set_ids
    )
    rows = cursor.fetchall()
    for row in rows:
        if row['filmed_at']:
            row['filmed_at'] = str(row['filmed_at'])
    return rows


def fetch_annotation_set_details(conn, annotation_set_ids):
    """Fetches annotation set provenance details from MySQL."""
    placeholders = ','.join(['%s'] * len(annotation_set_ids))
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        f"""SELECT id, frame_source, frames_extracted, sample_rate,
                   ls_project_name, ls_project_id, ls_min_task_id, ls_max_task_id, ls_downloaded_at
            FROM annotation_sets
            WHERE id IN ({placeholders})""",
        annotation_set_ids
    )
    return cursor.fetchall()


def log_video_sources(video_sources):
    """Serialises video metadata to a temp YAML, logs it as an artifact, then deletes the temp file."""
    path = 'video_sources.yaml'
    with open(path, 'w') as f:
        yaml.dump({'videos': video_sources}, f, default_flow_style=False)
    mlflow.log_artifact(path)
    os.remove(path)
    for v in video_sources:
        logger.info(f"{v['file_path']}  filmed={v['filmed_at']}  fish={v['fish_count']}  plants={v['plants']}")


def log_annotation_set_details(annotation_sets):
    """Serialises annotation set provenance to a temp YAML, logs it as an artifact, then deletes it."""
    path = 'annotation_sets.yaml'
    with open(path, 'w') as f:
        yaml.dump({'annotation_sets': annotation_sets}, f, default_flow_style=False)
    mlflow.log_artifact(path)
    os.remove(path)
    logger.info(f"annotation set details logged for ids={[a['id'] for a in annotation_sets]}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    run_path           = cfg['log_artifact_mlflow']['run_path']
    run_name           = run_path.strip('/').split('/')[-1]
    dataset_path       = f"dataset/{cfg['prepare_dataset']['dataset_name']}"
    annotation_set_ids = cfg['prepare_dataset']['annotation_set_ids']

    num_train = count_images(os.path.join(dataset_path, 'images', 'train'))
    num_val   = count_images(os.path.join(dataset_path, 'images', 'val'))

    logger.info(f"dataset path={dataset_path} train={num_train} val={num_val}")

    df = load_results_csv(run_path)

    mlflow.set_tracking_uri(MLRUNS_PATH)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param('yolo_model',          YOLO_MODEL)
        mlflow.log_param('dataset_name',        cfg['prepare_dataset']['dataset_name'])
        mlflow.log_param('dataset_folder',      dataset_path)
        mlflow.log_param('annotation_set_ids',  str(annotation_set_ids))
        mlflow.log_param('num_train',           num_train)
        mlflow.log_param('num_val',             num_val)
        logger.info(f"params logged: model={YOLO_MODEL} dataset={cfg['prepare_dataset']['dataset_name']} annotation_set_ids={annotation_set_ids} train={num_train} val={num_val}")

        log_epoch_metrics(df)
        log_dataset_card(dataset_path)

        with get_connection() as conn:
            video_sources      = fetch_video_sources(conn, annotation_set_ids)
            annotation_sets    = fetch_annotation_set_details(conn, annotation_set_ids)
        log_video_sources(video_sources)
        log_annotation_set_details(annotation_sets)

        mlflow.log_artifacts(run_path)

    logger.info("all metrics and artifacts logged — you can now delete the runs/ folder")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()


