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
import mlflow.pyfunc
import pandas as pd
import yaml


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

CONFIG_PATH  = 'config.yaml'
TRACKING_URI = 'sqlite:///mlflow.db'      # Model Registry needs a DB backend — the file-based mlruns/ store can't register models
EXPERIMENT   = 'aquamind'
YOLO_MODEL   = 'yolov8n'
MODEL_NAME   = 'aquamind-yolo-detector'   # the registered model; each retrain = a new auto-incremented version (v1, v2, ...)
ALIAS        = 'champion'                 # moving pointer to the current best version — downstream loads models:/<name>@champion


# ── MODEL WRAPPER ─────────────────────────────────────────────────────────────

class YoloModel(mlflow.pyfunc.PythonModel):
    """Wraps best.pt so MLflow logs it as a proper MODEL flavor (MLmodel + env → registrable + servable),
    instead of registering a bare .pt file. load_context rebuilds the native YOLO from the logged weights."""

    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts['weights'])

    def predict(self, context, model_input, params=None):
        results = self.model(model_input, verbose=False)
        return [r.boxes.data.cpu().numpy() for r in results]   # xyxy+conf+cls per image (generic serving path)


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


def load_dataset_card(dataset_path):
    """Loads dataset_card.yaml — the single source of truth for dataset provenance."""
    card_path = os.path.join(dataset_path, 'dataset_card.yaml')
    with open(card_path) as f:
        return yaml.safe_load(f)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    run_path     = cfg['log_artifact_mlflow']['run_path']
    run_name     = run_path.strip('/').split('/')[-1]
    dataset_path = f"dataset/{cfg['log_artifact_mlflow']['dataset_name']}"   # pinned to the run's own dataset, NOT prepare_dataset's (which points at the next dataset)

    card = load_dataset_card(dataset_path)

    num_train = count_images(os.path.join(dataset_path, 'images', 'train'))
    num_val   = count_images(os.path.join(dataset_path, 'images', 'val'))

    logger.info(f"dataset path={dataset_path} train={num_train} val={num_val}")

    df = load_results_csv(run_path)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param('yolo_model',          YOLO_MODEL)
        mlflow.log_param('dataset_name',        card['dataset_name'])
        mlflow.log_param('dataset_folder',      dataset_path)
        mlflow.log_param('annotation_set_ids',  str(card['annotation_set_ids']))
        mlflow.log_param('num_train',           num_train)
        mlflow.log_param('num_val',             num_val)
        mlflow.log_param('git_commit',          card['git_commit'])
        logger.info(f"params logged: model={YOLO_MODEL} dataset={card['dataset_name']} annotation_set_ids={card['annotation_set_ids']} train={num_train} val={num_val}")

        log_epoch_metrics(df)
        log_dataset_card(dataset_path)
        mlflow.log_artifacts(run_path)

        # ── MODEL REGISTRY (textbook) — log best.pt as a proper pyfunc MODEL flavor + register in one call ──
        # log_model creates an MLmodel flavor (servable + loadable via models:/), registered_model_name= registers
        # it atomically as a new VERSION linked to THIS run's lineage. Then move the @champion alias to it.
        best_pt = os.path.join(run_path, 'weights', 'best.pt')
        info = mlflow.pyfunc.log_model(
            name='model',
            python_model=YoloModel(),
            artifacts={'weights': best_pt},
            registered_model_name=MODEL_NAME,
            pip_requirements=['ultralytics', 'torch'],
        )
        version = info.registered_model_version                          # v1, v2, ...
        mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version)
        logger.info(f"registered '{MODEL_NAME}' v{version} (alias @{ALIAS}) — lineage: dataset={card['dataset_name']}, commit={card['git_commit'][:8]}")

    logger.info("metrics + artifacts logged and model registered — you can now delete the runs/ folder")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
