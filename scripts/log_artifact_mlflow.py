# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os

import mlflow
import pandas as pd
import yaml

from scripts.db import get_connection


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'
MLRUNS_PATH = '/Users/govinda-dashugolienart/Documents/Github_HD/AquaMind/mlruns'
EXPERIMENT  = 'aquamind'
YOLO_MODEL  = 'yolov8s'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.png')])


def load_results_csv(run_path):
    df = pd.read_csv(os.path.join(run_path, 'results.csv'))
    df.columns = df.columns.str.strip()
    return df


def log_epoch_metrics(df):
    for _, row in df.iterrows():
        epoch = int(row['epoch'])
        mlflow.log_metrics({
            'train/box_loss': row['train/box_loss'],
            'train/dfl_loss': row['train/dfl_loss'],
            'train/cls_loss': row['train/cls_loss'],
            'val/box_loss':   row['val/box_loss'],
            'val/dfl_loss':   row['val/dfl_loss'],
            'val/cls_loss':   row['val/cls_loss'],
            'precision':      row['metrics/precision(B)'],
            'recall':         row['metrics/recall(B)'],
            'mAP50':          row['metrics/mAP50(B)'],
            'mAP50_95':       row['metrics/mAP50-95(B)'],
            'lr0':            row['lr/pg0'],
            'lr1':            row['lr/pg1'],
            'lr2':            row['lr/pg2'],
        }, step=epoch)
    print(f"  {len(df)} epochs logged")


def log_dataset_card(dataset_path):
    card_path = os.path.join(dataset_path, 'dataset_card.yaml')
    if os.path.exists(card_path):
        mlflow.log_artifact(card_path)
        print(f"  Dataset card logged from {card_path}")
    else:
        print(f"  No dataset_card.yaml found in {dataset_path} — skipping.")


def fetch_video_sources(conn, session_prefix):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT DISTINCT v.file_path, v.filmed_at, v.session_type, v.obstacles, v.fish_count, v.notes
        FROM videos v
        JOIN frames fr ON fr.video_id = v.id
        JOIN annotations a ON a.frame_id = fr.id
        WHERE a.session_id LIKE %s
    """, (f"{session_prefix}%",))
    rows = cursor.fetchall()
    for row in rows:
        if row['filmed_at']:
            row['filmed_at'] = str(row['filmed_at'])
    return rows


def log_video_sources(video_sources):
    path = 'video_sources.yaml'
    with open(path, 'w') as f:
        yaml.dump({'videos': video_sources}, f, default_flow_style=False)
    mlflow.log_artifact(path)
    os.remove(path)
    for v in video_sources:
        print(f"  {v['file_path']}  filmed={v['filmed_at']}  fish={v['fish_count']}  obstacles={v['obstacles']}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    run_path       = cfg['log_artifact_mlflow']['run_path']
    run_name       = run_path.strip('/').split('/')[-1]
    dataset_path   = f"dataset/{cfg['prepare_dataset']['dataset_name']}"
    session_prefix = cfg['prepare_dataset']['annotation_session_id']

    num_train = count_images(os.path.join(dataset_path, 'images', 'train'))
    num_val   = count_images(os.path.join(dataset_path, 'images', 'val'))

    print("─" * 50)
    print("  DATASET")
    print("─" * 50)
    print(f"  Path  : {dataset_path}")
    print(f"  Train : {num_train} images")
    print(f"  Val   : {num_val} images")

    df = load_results_csv(run_path)

    mlflow.set_tracking_uri(MLRUNS_PATH)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name):

        print("─" * 50)
        print("  PARAMETERS")
        print("─" * 50)
        mlflow.log_param('yolo_model',     YOLO_MODEL)
        mlflow.log_param('dataset_folder', dataset_path)
        mlflow.log_param('num_train',      num_train)
        mlflow.log_param('num_val',        num_val)
        print(f"  model={YOLO_MODEL}  train={num_train}  val={num_val}")

        print("─" * 50)
        print("  METRICS")
        print("─" * 50)
        log_epoch_metrics(df)

        print("─" * 50)
        print("  ARTIFACTS")
        print("─" * 50)
        log_dataset_card(dataset_path)

        print("─" * 50)
        print("  VIDEO SOURCES")
        print("─" * 50)
        with get_connection() as conn:
            video_sources = fetch_video_sources(conn, session_prefix)
        log_video_sources(video_sources)

        mlflow.log_artifacts(run_path)

    print("─" * 50)
    print("  DONE")
    print("─" * 50)
    print("  All metrics and artifacts logged. You can now delete the runs/ folder.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
