import mlflow
import os
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
LOCAL_MLRUNS = '/Users/govinda-dashugolienart/Documents/Github_HD/AquaMind/mlruns'
LOCAL_RUNS   = '/Users/govinda-dashugolienart/Documents/Github_HD/AquaMind/runs'

mlflow.set_tracking_uri(LOCAL_MLRUNS)
mlflow.set_experiment('aquamind')

# ── User input ───────────────────────────────────────────────────────────────
run_path      = input("Enter relative path to run folder (e.g. runs/aquamind_run34): ")
run_name      = run_path.strip('/').split('/')[-1]
dataset_path  = input("Enter relative path to dataset folder (e.g. dataset/IMG_0350_...): ")

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

    mlflow.log_param('dataset_folder', dataset_path)
    mlflow.log_param('num_train', num_train)
    mlflow.log_param('num_val', num_val)

    for _, row in df.iterrows():
        epoch = int(row['epoch'])
        mlflow.log_metrics({
            'train/box_loss': row['train/box_loss'],
            'train/obj_loss': row['train/obj_loss'],
            'train/cls_loss': row['train/cls_loss'],
            'val/box_loss':   row['val/box_loss'],
            'val/obj_loss':   row['val/obj_loss'],
            'val/cls_loss':   row['val/cls_loss'],
            'precision':      row['metrics/precision'],
            'recall':         row['metrics/recall'],
            'mAP50':          row['metrics/mAP_0.5'],
            'mAP50_95':       row['metrics/mAP_0.5:0.95'],
            'lr0':            row['x/lr0'],
            'lr1':            row['x/lr1'],
            'lr2':            row['x/lr2'],
        }, step=epoch)

    # ── Log all artifacts (images, yaml files, weights) ───────────────────────
    run_folder = run_path
    mlflow.log_artifacts(run_folder)

print("All metrics and artifacts logged. You can now delete the runs/ folder.")
