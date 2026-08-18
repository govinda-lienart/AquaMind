"""usage:  python -m scripts.train_chase_classifier
hardcoded path to train_df/test_df built by build_chase_windows.py"""


# IMPORTS

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # avoids popup windows of produced plots
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name

# CONSTANTS

VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
RANDOM_SEED = 42

# HELPERS - shared between every model tried below, so each model's block is just 3 calls

def evaluate_model(model, model_name, X_test, y_test, output_folder):
    """predict on X_test, print classification_report, save confusion matrix png"""
    banner_sub(f'{model_name} - PREDICT + SCORE on X_test')
    y_pred = model.predict(X_test)
    logger.info(f'\n{classification_report(y_test, y_pred, target_names=["not_chase", "chase"])}')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=['not_chase', 'chase']).plot(ax=ax)
    ax.set_title(model_name)
    path_plot = os.path.join(output_folder, f'{model_name}_confusion_matrix.png')
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'saved confusion matrix -> {path_plot}')


def plot_feature_importance(values, feature_cols, model_name, xlabel, output_folder):
    """values = model.coef_[0] (LR) or model.feature_importances_ (RF) - same plot shape either way"""
    banner_sub(f'{model_name} - FEATURE IMPORTANCE PLOT')
    logger.info(f'{model_name} feature importance: {dict(zip(feature_cols, values))}')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_cols, values)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(model_name)
    path_plot = os.path.join(output_folder, f'{model_name}_feature_importance.png')
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'saved feature importance -> {path_plot}')


# MAIN

# STEP 1 - load train_df / test_df saved by build_chase_windows.py
banner('STEP 1 - LOAD train_df / test_df')
parquet_path, *_ = grab_video_name(VIDEO_RUN_NAME) # i don't need to unpack pixels_per_cm, calibration_secs, surface_y_px, so is a throwaway list *_
split_folder = os.path.join(os.path.dirname(parquet_path), 'chase_train_test') #output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet -> output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/chase_train_test
train_df = pd.read_parquet(os.path.join(split_folder, 'train_df.parquet'))
test_df = pd.read_parquet(os.path.join(split_folder, 'test_df.parquet'))
logger.info(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

# STEP 2 - split each X (features) and y (label) - no need to keep event_id
banner('STEP 2 - SPLIT X / y')
feature_cols = [c for c in train_df.columns if c not in ('event_id', 'label', 'fish_id_a', 'fish_id_b', 'chaser_id', 'window_frame_start', 'window_frame_end')] # builds a list of column names to use as features ( but doesnt use event_id, label, or the identifier/bookkeeping columns) - acutally i could have added the column manually in the menu but this way cleaner but advantage is that it autoadjust...i dont need to change in case i add more feautures
logger.info(f'feature_cols: {feature_cols}')

X_train = train_df[feature_cols] # features of train
y_train = train_df['label'] # label of train
X_test = test_df[feature_cols] # features of test
y_test = test_df['label'] # label of test
logger.info(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
logger.info(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

output_folder = os.path.join(split_folder, 'output_train_chase_classifier')
os.makedirs(output_folder, exist_ok=True)

# STEP 3 - MODEL A: LogisticRegression (baseline #1)
banner('STEP 3 - MODEL A: LOGISTIC REGRESSION')
banner_sub('LOGISTIC REGRESSION - TRAIN')
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)  # trains
logger.info('model trained')

evaluate_model(logreg_model, 'logistic_regression', X_test, y_test, output_folder)
plot_feature_importance(
    logreg_model.coef_[0],  # one weight per feature, in the same order as feature_cols
    feature_cols, 'logistic_regression',
    'coefficient (pushes toward chase if positive, not_chase if negative)',
    output_folder,
)

# STEP 4 - MODEL B: RandomForest (baseline #2)
banner('STEP 4 - MODEL B: RANDOM FOREST')
banner_sub('RANDOM FOREST - TRAIN')
rf_model = RandomForestClassifier(random_state=RANDOM_SEED)
rf_model.fit(X_train, y_train)
logger.info('model trained')

evaluate_model(rf_model, 'random_forest', X_test, y_test, output_folder)
plot_feature_importance(
    rf_model.feature_importances_,  # always >= 0, no direction (unlike LR coefficients)
    feature_cols, 'random_forest',
    'importance (higher = model relied on this feature more, no direction)',
    output_folder,
)

# STEP 5 - pull out the misclassified test windows (LR and RF agreed on the same 3 mistage cases) for manual video review
banner('STEP 5 - MISCLASSIFIED WINDOWS (for manual video review)')
y_pred_rf = rf_model.predict(X_test)  # same as logreg_model's predictions on this run - both models agreed
mismatches = test_df[y_test.values != y_pred_rf].copy()
mismatches['predicted_label'] = y_pred_rf[y_test.values != y_pred_rf]
logger.info(f'{mismatches.shape[0]} misclassified windows out of {test_df.shape[0]} test windows')

review_cols = ['event_id', 'label', 'predicted_label', 'fish_id_a', 'fish_id_b', 'chaser_id', 'window_frame_start', 'window_frame_end']
logger.info(f'\n{mismatches[review_cols].to_string()}')

""" 3 misclassified windows out of 40 test windows
    event_id  label  predicted_label  fish_id_a  fish_id_b  window_frame_start  window_frame_end
0          0      1                0          1          3                 631               665 # checked the video - that is defineitly a chasing event
18        27      0                1          1          4                4084              4118 # here the fish is heading towards the supervicia as a burst and going near another fish but this is not an attack...actually the other fish does burst escape but it could look like a chasing.
19        27      0                1          1          4                4101              4135

"""