"usage: "

# IMPORTS

import pandas as pd
import numpy as np
import os
from scripts.video_utils import grab_video_name, trim_to_calibration
from scripts.console import banner, banner_sub
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'

# MAIN

# Step 1 — Load train_df/test_df
banner("Step 1 — Load train_df/test_df")
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME )
output_folder = os.path.dirname(parquet_path)
feeding_train_test_path = os.path.join(output_folder, "feeding_train_test")
train_parquet_path = os.path.join(feeding_train_test_path, "train_df.parquet")
test_parquet_path = os.path.join(feeding_train_test_path, "test_df.parquet")
train_df = pd.read_parquet(train_parquet_path)
test_df = pd.read_parquet(test_parquet_path)
logger.info(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

# Step 2 — split into X (features) / y (label)
banner("Step 2 — split into X (features) / y (label)")
FEATURE_COLS = ["mean_speed_cm_s", "max_speed_cm_s", "mean_burst", "max_burst"]

X_train = train_df[FEATURE_COLS]
y_train = train_df["label"]
X_test = test_df[FEATURE_COLS]
y_test = test_df["label"]
logger.info(f"Xtrain: {X_train.shape[0]},y_train: {y_train.shape[0]}")
logger.info(f"Xtest: {X_test.shape[0]},y_test: {y_test.shape[0]}")

# STEP 3— LOGISTIC REGRESSION
banner("STEP 3 - LogisticRegression")

    # set up a dedicated, timestamped output folder for this script's plots
stamp = datetime.now().strftime('%Y_%m_%d_%H%M')
classifier_output_folder = os.path.join(feeding_train_test_path, "output_train_feeding_classifier", stamp)
os.makedirs(classifier_output_folder, exist_ok=True)

    # training
model_name = "logistic_regression"
logreg_model = LogisticRegression()

    # training step - learn from data - uses y_train (labels) directly to adjust the model's internal weights
logreg_model.fit(X_train, y_train)
logger.info('model trained')

    # testing generalization - asking the trained model to guess the label for data it has never seen before
y_pred = logreg_model.predict(X_test)

    # predicting on the same rows it was trained on, to compare against y_pred and catch overfitting
train_pred = logreg_model.predict(X_train)
logger.info(f'\n{classification_report(y_test, y_pred, target_names=["not_feeding", "feeding"])}')
            # Precision: out of every window the model predicted as feeding, what fraction actually were.
            # Recall: out of every window that was actually feeding, what fraction the model caught.

logger.info(f'train accuracy: {(train_pred == y_train).mean():.2f}, test accuracy: {(y_pred == y_test).mean():.2f}') #train accuracy: 0.57, test accuracy: 0.62 # overfitting definitely isn't happening — if the model were memorizing training data, train accuracy would be inflated above test, not below it.

    # confusion matrix
cm = confusion_matrix(y_test, y_pred) # Compares your real test labels (y_test) against the model's predictions (y_pre
fig, ax = plt.subplots(figsize=(5, 5)) # fig is the whole image/canvas, ax is the actual plotting area #  5×5 inch square - good for gridss
ConfusionMatrixDisplay(cm, display_labels=["not_feeding", "feeding"]).plot(ax=ax) 
                                       # Takes the raw cm numbers and actually draws them as a colored grid 
                                       # Display_labels tells it what to write as the row/column labels 
                                       # .plot(ax=ax) draws it onto the ax
ax.set_title(model_name)
cm_path = os.path.join(classifier_output_folder, f"{model_name}_confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight") # dpi=150 controls resolution/sharpness (higher = crisper but bigger file). bbox_inches="tight" trims excess white space around the plot 
plt.close(fig)
logger.info(f'saved confusion matrix -> {cm_path}')

    # feature importance plot / coefficient plot 
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(FEATURE_COLS, logreg_model.coef_[0])
            #barh() draws a horizontal bar chart (bars extending left-right, not up-down) 
            # — FEATURE_COLS (your list of 4 feature names) becomes the labels on the y-axis, one per bar
            # and logreg_model.coef_[0] provides each bar's length/value. .coef_ is where sklearn stores the model's learned weights after training — it's technically a 2D array (shape (1, 4) here, since this is binary classification), so [0] grabs the single row of 4 actual coefficient numbers out of it.
ax.set_xlabel("coefficient (pushes toward feeding if positive, not_feeding if negative)") # Labels the x-axis 
ax.axvline(0, color="black", linewidth=0.8) # # Draws a vertical black line at x=0 
importance_path = os.path.join(classifier_output_folder, f"{model_name}_feature_importance.png")
fig.savefig(importance_path, dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info(f'saved feature importance -> {importance_path}')