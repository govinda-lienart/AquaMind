"""
Logs a YOLO training run to MLflow and registers its best.pt as a new model version.

Input  : run_path, dataset_name, alias from config.yaml (log_artifact_mlflow section)
Needs  : the run folder in runs/ (downloaded from Kaggle), the dataset folder with its dataset_card.yaml
Output : an MLflow run (params, per-epoch metrics, artifacts) + a new version of
         aquamind-yolo-detector with the chosen alias (@baseline, @champion, ...)

Usage: python -m scripts.log_artifact_mlflow
"""

# IMPORTS───────────────────────────────────────────

from console import banner, banner_sub

# CONSTANTS──────────────────────────────────────────



# ── STEP 1: READ config.yaml ──────────────────────────────────────────────────
banner("STEP 1 - READ config.yaml")


# ── STEP 2: LOAD dataset_card.yaml ────────────────────────────────────────────
banner("STEP 2 - LOAD dataset_card.yaml")


# ── STEP 3: COUNT train / val images ──────────────────────────────────────────
banner("STEP 3 - COUNT train / val images")


# ── STEP 4: LOAD results.csv + args.yaml ──────────────────────────────────────
banner("STEP 4 - LOAD results.csv + args.yaml")


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