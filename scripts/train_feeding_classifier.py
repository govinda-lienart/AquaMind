"""
- Trains classifiers: LogisticRegression, RandomForest, XGBoost
- Uses windows built by build_feeding_windows_fixed_window.py
- Saves a confusion matrix + feature-importance plot per model

usage:
    conda activate aquamind
    python -m scripts.train_feeding_classifier
"""

# IMPORTS

import pandas as pd
import numpy as np
import os

from scripts.video_utils import grab_video_name, trim_to_calibration
from scripts.console import banner, banner_sub
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
FEATURE_COLS = ["mean_speed_cm_s", "max_speed_cm_s", "mean_burst", "max_burst"]
CLASS_NAMES = ["not_feeding", "feeding"]

# HELPERS

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, output_folder):
    # creates confusion matrix
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    logger.info(f'\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}')
    logger.info(
        f'train accuracy: {(train_pred == y_train).mean():.2f}, '
        f'test accuracy: {(y_pred == y_test).mean():.2f}'
    )

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax)
    ax.set_title(model_name)
    cm_path = os.path.join(output_folder, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f'saved confusion matrix -> {cm_path}')

    return y_pred

def plot_importance(values, model_name, xlabel, output_folder, draw_zero_line=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(FEATURE_COLS, values)
    ax.set_xlabel(xlabel)
    if draw_zero_line:
        ax.axvline(0, color="black", linewidth=0.8)
    importance_path = os.path.join(output_folder, f"{model_name}_feature_importance.png")
    fig.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f'saved feature importance -> {importance_path}')

# MAIN

# Step 1 — Load train_df/test_df
banner("Step 1 — Load train_df/test_df")
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME)
output_folder = os.path.dirname(parquet_path)
feeding_train_test_path = os.path.join(output_folder, "feeding_train_test")
train_df = pd.read_parquet(os.path.join(feeding_train_test_path, "train_df.parquet"))
test_df = pd.read_parquet(os.path.join(feeding_train_test_path, "test_df.parquet"))
logger.info(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

# Step 2 — split into X (features) / y (label)
banner("Step 2 — split into X (features) / y (label)")
X_train = train_df[FEATURE_COLS]
y_train = train_df["label"]
X_test = test_df[FEATURE_COLS]
y_test = test_df["label"]
logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# dedicated, timestamped output folder for this run's plots
stamp = datetime.now().strftime('%Y_%m_%d_%H%M')
classifier_output_folder = os.path.join(feeding_train_test_path, "output_train_feeding_classifier", stamp)
os.makedirs(classifier_output_folder, exist_ok=True)

# Step 3 — MODEL A: LogisticRegression (one linear boundary)
banner("Step 3 — LogisticRegression")
logreg_model = LogisticRegression().fit(X_train, y_train)
evaluate_model(logreg_model, "logistic_regression", X_train, y_train, X_test, y_test, classifier_output_folder)
plot_importance(
    logreg_model.coef_[0], "logistic_regression",
    "coefficient (pushes toward feeding if positive, not_feeding if negative)",
    classifier_output_folder, draw_zero_line=True,
)

# Step 4 — MODEL B: RandomForest (100 parallel trees, majority vote)
banner("Step 4 — RandomForest")
rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
evaluate_model(rf_model, "random_forest", X_train, y_train, X_test, y_test, classifier_output_folder)
plot_importance(
    rf_model.feature_importances_, "random_forest",
    "feature importance (higher = model relies on it more)",
    classifier_output_folder,
)

# Step 5 — MODEL C: XGBoost (sequential boosted trees, each fixes the last one's errors)
banner("Step 5 — XGBoost")
xgb_model = XGBClassifier(random_state=42).fit(X_train, y_train)
evaluate_model(xgb_model, "xgboost", X_train, y_train, X_test, y_test, classifier_output_folder)
plot_importance(
    xgb_model.feature_importances_, "xgboost",
    "feature importance (higher = model relies on it more)",
    classifier_output_folder,
)
