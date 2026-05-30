import mlflow
import os
import pandas as pd
import yaml

# ── Configuration ────────────────────────────────────────────────────────────-
LOCAL_MLRUNS = '/Users/govinda-dashugolienart/Documents/Github_HD/AquaMind/mlruns'
LOCAL_RUNS   = '/Users/govinda-dashugolienart/Documents/Github_HD/AquaMind/runs'

mlflow.set_tracking_uri(LOCAL_MLRUNS)
mlflow.set_experiment('aquamind')

# ── User input ───────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
run_path      = cfg['log_artifact_mlflow']['run_path']
run_name      = run_path.strip('/').split('/')[-1]
dataset_path  = f"dataset/{cfg['prepare_dataset']['dataset_name']}"
yolo_model    = "yolov8s"

train_path = os.path.join(dataset_path, 'images', 'train')
val_path   = os.path.join(dataset_path, 'images', 'val')
num_train  = len([f for f in os.listdir(train_path) if f.endswith('.png')])
num_val    = len([f for f in os.listdir(val_path)   if f.endswith('.png')])

print(f"Dataset : {dataset_path}")
print(f"Train   : {num_train} images")
print(f"Val     : {num_val} images")

# ── Log metrics from results.csv ─────────────────────────────────────────────
results_path = os.path.join(run_path, 'results.csv')
df = pd.read_csv(results_path)
df.columns = df.columns.str.strip()

with mlflow.start_run(run_name=run_name):

    mlflow.log_param('yolo_model',     yolo_model)
    mlflow.log_param('dataset_folder', dataset_path)
    mlflow.log_param('num_train',      num_train)
    mlflow.log_param('num_val',        num_val)

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

    # ── Log dataset card if present ───────────────────────────────────────────
    dataset_card_path = os.path.join(dataset_path, 'dataset_card.yaml')
    if os.path.exists(dataset_card_path):
        with open(dataset_card_path, 'r') as f:
            card = yaml.safe_load(f)
        mlflow.log_param('dvc_hash',   card.get('dvc_hash', ''))
        mlflow.log_param('sql_query',  card.get('sql_query', ''))
        mlflow.log_param('notes',      card.get('notes', ''))
        mlflow.log_param('sources',    str(card.get('sources', '')))
        mlflow.log_artifact(dataset_card_path)
        print(f"Dataset card logged from {dataset_card_path}")
    else:
        print(f"No dataset_card.yaml found in {dataset_path} — skipping.")

    # ── Log all artifacts (images, yaml files, weights) ───────────────────────
    run_folder = run_path
    mlflow.log_artifacts(run_folder)

print("All metrics and artifacts logged. You can now delete the runs/ folder.")
