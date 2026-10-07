"""
Logs a YOLO training run to MLflow and registers its best.pt as a new model version.

Input  : run_path, dataset_name, alias from config.yaml (log_artifact_mlflow section)
Needs  : the run folder in runs/ (downloaded from Kaggle), the dataset folder with its dataset_card.yaml
Output : an MLflow run (params, per-epoch metrics, artifacts) + a new version of
         aquamind-yolo-detector with the chosen alias (@baseline, @champion, ...)

Usage: python -m scripts.log_artifact_mlflow
"""

# IMPORTS───────────────────────────────────────────

# libraries
import os                
import yaml               
import mlflow #  tracking: run, params, metrics, artifacts, alias.
import mlflow.pyfunc # the model wrapper: - YoloModel subclasses mlflow.pyfunc.PythonModel
import pandas as pd

# module imports
from scripts.console import banner

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS──────────────────────────────────────────
TRACKING_URI = "sqlite:///mlflow.db"   # registry needs a database backend (the mlruns/ file store can't register models)
EXPERIMENT   = "aquamind_detector"              # the MLflow experiment every YOLO detector run is logged under
MODEL_NAME   = "aquamind-yolo-detector"   # the registered model: each logged run adds a new version (v1, v2, ...)

# MODEL WRAPPER──────────────────────────────────────

class YoloModel(mlflow.pyfunc.PythonModel):
    """Wraps best.pt as a proper MLflow model flavor, so the registry holds a real model, not a bare file."""

    # instructions for anyone loading this model through MLflow (not really used here by the tracker, which loads best.pt directly)
    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])          # rebuild YOLO from the logged best.pt

    # runs when someone asks for predictions: images in, boxes out
    def predict(self, context, model_input, params=None):
        results = self.model(model_input, verbose=False)
        return [r.boxes.data.cpu().numpy() for r in results]     # boxes per image: x1, y1, x2, y2, conf, class

# ── STEP 1: READ config.yaml ──────────────────────────────────────────────────
banner("STEP 1 - READ config.yaml")

# 1a. load the log_artifact_mlflow section of config.yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)                  
cfg = cfg["log_artifact_mlflow"]            

# 1b. pull out run_path, dataset_name and alias
run_path     = cfg["run_path"]              
dataset_name = cfg["dataset_name"]          
alias        = cfg["alias"]                 
run_name     = os.path.basename(run_path)    
logger.info(f"run: {run_name} | dataset: {dataset_name} | alias: @{alias}")

# ── STEP 2: LOAD dataset_card.yaml ────────────────────────────────────────────
banner("STEP 2 - LOAD dataset_card.yaml")
# 2a. building the dataset folder path from the pinned dataset_name
dataset_path = os.path.join("dataset", dataset_name)        # e.g dataset/regular_data_ann_5r_9r_10r_2026_10_01_11h47_2026_10_06_09h48
card_path    = os.path.join(dataset_path, "dataset_card.yaml")

# 2b. load the card: the provenance record prepare_dataset wrote (sets, videos, counts, git commit)
with open(card_path) as f:
    card = yaml.safe_load(f)
logger.info(f"card loaded: {card['dataset_name']} | sets {card['annotation_set_ids']} | commit {card['git_commit']}")

# ── STEP 3: COUNT train / val images ──────────────────────────────────────────
banner("STEP 3 - COUNT train / val images")
# 3a. count the images actually on disk (.jpg / .png) in each split folder
train_dir = os.path.join(dataset_path, "images", "train")
val_dir   = os.path.join(dataset_path, "images", "val")
num_train = len([f for f in os.listdir(train_dir) if f.endswith((".jpg", ".png"))])
num_val   = len([f for f in os.listdir(val_dir)   if f.endswith((".jpg", ".png"))])
logger.info(f"on disk: {num_train} train | {num_val} val")

# 3b. compare with what the card says: a mismatch means frames were lost while copying
if (num_train, num_val) != (card["num_train"], card["num_val"]): # comparing tuples
    logger.warning(f"MISMATCH: card says {card['num_train']} train / {card['num_val']} val, disk has {num_train} / {num_val}")

# ── STEP 4: LOAD results.csv + args.yaml ──────────────────────────────────────
banner("STEP 4 - LOAD results.csv + args.yaml")
# 4a. results.csv: one row per epoch (losses, precision, recall, mAP, learning rates)
results = pd.read_csv(os.path.join(run_path, "results.csv"))
results.columns = results.columns.str.strip() # if ['                  epoch', ... -> repairs to ['epoch', 
logger.info(f"results.csv: {len(results)} epochs") 

# 4a (extra). quick look at the scores: best epoch vs last epoch
best = results.loc[results["metrics/mAP50-95(B)"].idxmax()] # .idxmax() → "which index has the highest score e.g 19 -->  results.loc[19]  give me that whole row" contrast with .iloc which looks by position
last = results.iloc[-1]                                     # the last row (final epoch), by position
logger.info(f"best epoch {int(best['epoch'])}: mAP50 {best['metrics/mAP50(B)']:.3f} | mAP50-95 {best['metrics/mAP50-95(B)']:.3f}")
logger.info(f"last epoch {int(last['epoch'])}: mAP50 {last['metrics/mAP50(B)']:.3f} | mAP50-95 {last['metrics/mAP50-95(B)']:.3f}")

# 4b. args.yaml: the exact settings YOLO trained with (written by YOLO itself)
with open(os.path.join(run_path, "args.yaml")) as f:
    train_args = yaml.safe_load(f)
yolo_model = train_args["model"].replace(".pt", "")   # e.g "yolov8s.pt" → "yolov8s"
logger.info(f"trained: {yolo_model} | {train_args['epochs']} epochs | imgsz {train_args['imgsz']} | batch {train_args['batch']}")

# ── STEP 5: START MLflow run + LOG params ─────────────────────────────────────
banner("STEP 5 - START MLflow run + LOG params")
# 5a. connect to the MLflow database and pick the experiment
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# 5b. start the run: everything logged until mlflow.end_run() (end of STEP 8) belongs to it
mlflow.start_run(run_name=run_name)

# 5c. log params: what was trained, on which data, with which code
mlflow.log_param("yolo_model",         yolo_model)                    # from args.yaml (not hardcoded)
mlflow.log_param("epochs",             train_args["epochs"])
mlflow.log_param("imgsz",              train_args["imgsz"])
mlflow.log_param("batch",              train_args["batch"])
mlflow.log_param("dataset_name",       card["dataset_name"])
mlflow.log_param("annotation_set_ids", str(card["annotation_set_ids"]))
mlflow.log_param("num_train",          num_train)
mlflow.log_param("num_val",            num_val)
mlflow.log_param("git_commit",         card["git_commit"])            # commit of the code that BUILT the dataset
logger.info(f"params logged: {yolo_model} | {card['dataset_name']} | sets {card['annotation_set_ids']} | {num_train} train / {num_val} val")

# ── STEP 6: LOG per-epoch metrics ─────────────────────────────────────────────
banner("STEP 6 - LOG per-epoch metrics")
# 6a. map YOLO's column names to clean MLflow metric names

METRIC_NAMES = {
    "train/box_loss":       "train/box_loss",
    "train/cls_loss":       "train/cls_loss",
    "train/dfl_loss":       "train/dfl_loss",
    "val/box_loss":         "val/box_loss",
    "val/cls_loss":         "val/cls_loss",
    "val/dfl_loss":         "val/dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)":    "recall",
    "metrics/mAP50(B)":     "mAP50",
    "metrics/mAP50-95(B)":  "mAP50_95",
    "lr/pg0":               "lr0",
    "lr/pg1":               "lr1",
    "lr/pg2":               "lr2",
}

# 6b. send each epoch's scores to MLflow; step=epoch is the x-axis position (so MLflow can draw a curve)
for index, row in results.iterrows():                    # for each epoch (50 times)
    epoch   = int(row["epoch"])                          # epoch comes from the ROW, not from the index
    metrics = {}                                         # 1. start with an empty dictionary then # {'train/box_loss': 1.95943, 'train/cls_loss': 1.21242, 'train/dfl_loss': 1.29453,....
    for yolo_name, mlflow_name in METRIC_NAMES.items():  # The inner loop (for yolo_name, mlflow_name in METRIC_NAMES.items()) goes, within that one epoch, through each value, 13 times: box loss, cls loss @         # 2. walk through the translation table, pair by pair (13 times)
        value = row[yolo_name]                           # 3. read the value from this epoch's row (by YOLO's name) # xrow["metrics/mAP50(B)"] → 0.6135.
        metrics[mlflow_name] = value                     # 4. store it under the MLflow name mtrics = {}
    mlflow.log_metrics(metrics, step=epoch)              # dictionary full: send this epoch's 13 values, placed at x = epoch
logger.info(f"{len(results)} epochs of metrics logged")  # after the loop: runs once

# ── STEP 7: LOG artifacts ─────────────────────────────────────────────────────
banner("STEP 7 - LOG artifacts")
# 7a. the dataset card: which batches, videos and code commit the training data came from
mlflow.log_artifact(card_path)
logger.info(f"dataset card logged: {card_path}")

# 7b. the whole run folder: results.csv, args.yaml, plots (results.png, confusion matrix, PR curves), weights/
mlflow.log_artifacts(run_path)
logger.info(f"run folder logged: {run_path}")

# 7b. the whole run folder: results.csv, args.yaml, plots (results.png, confusion matrix, PR curves), weights/
mlflow.log_artifacts(run_path)
logger.info(f"run folder logged: {run_path}")


# ── STEP 8: REGISTER model + SET alias ────────────────────────────────────────
banner("STEP 8 - REGISTER model + SET alias")

# 8a. packs the existing best.pt (plus instructions), --- >log best.pt as an MLflow model AND register it as a new version of MODEL_NAME, in one call
best_pt = os.path.join(run_path, "weights", "best.pt")
info = mlflow.pyfunc.log_model( # this line packs the model and puts it on the shelf in the registry.
    name="model",                              # the package's name (stored in mlruns/<exp>/models/)
    python_model=YoloModel(),                  # the wrapper (instructions for loading / predicting)
    artifacts={"weights": best_pt},            # the file packed with it → context.artifacts["weights"]
    registered_model_name=MODEL_NAME,          # also register it as a new version in the registry
    pip_requirements=["ultralytics", "torch"], # what someone needs installed to load it
)
version = info.registered_model_version        # e.g 5 → the number this version got

# 8b. move the alias from config.yaml (baseline / champion) to this new version
mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)
logger.info(f"registered {MODEL_NAME} v{version} → @{alias}")

# 8c. close the run opened in STEP 5
mlflow.end_run()

# ── STEP 9: SUMMARY ───────────────────────────────────────────────────────────
banner("STEP 9 - SUMMARY")

# 9a. what was logged and registered, in one glance
logger.info(f"run          : {run_name}")
logger.info(f"dataset      : {card['dataset_name']} (sets {card['annotation_set_ids']})")
logger.info(f"trained      : {yolo_model} | {num_train} train / {num_val} val")
logger.info(f"registered   : {MODEL_NAME} v{version} → @{alias}")
logger.info(f"load it with : models:/{MODEL_NAME}@{alias}")