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
from scripts.console import banner, banner_sub

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS──────────────────────────────────────────



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
# 4a (extra). quick look at the scores: best epoch vs last epoch
results = pd.read_csv(os.path.join(run_path, "results.csv"))
results.columns = results.columns.str.strip() # if ['                  epoch', ... -> repairs to ['epoch', 
logger.info(f"results.csv: {len(results)} epochs") 
best = results.loc[results["metrics/mAP50-95(B)"].idxmax()] # .idxmax() → "which index has the highest score e.g 19 -->  results.loc[19]  give me that whole row" contrast with .iloc which looks by position

# 4b. args.yaml: the exact settings YOLO trained with (written by YOLO itself)
with open(os.path.join(run_path, "args.yaml")) as f:


# ── STEP 5: START MLflow run + LOG params ─────────────────────────────────────
banner("STEP 5 - START MLflow run + LOG params")


# ── STEP 6: LOG per-epoch metrics ─────────────────────────────────────────────
banner("STEP 6 - LOG per-epoch metrics")


# ── STEP 7: LOG artifacts ─────────────────────────────────────────────────────
banner("STEP 7 - LOG artifacts")


# ── STEP 8: REGISTER model + SET alias ────────────────────────────────────────
banner("STEP 8 - REGISTER model + SET alias")


# ── STEP 9: SUMMARY ───────────────────────────────────────────────────────────
banner("STEP 9 - SUMMARY")