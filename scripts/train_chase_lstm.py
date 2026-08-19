# IMPORTS

import os
import argparse
from datetime import datetime
import yaml
import pandas as pd
from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, trim_to_calibration, build_pairs, build_sequences
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANT
LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
WINDOW_SIZE_FRAMES = 35
STRIDE_FRAMES = 17
MAX_WINDOWS_PER_NEGATIVE_EVENT = 4
RANDOM_SEED = 42


# MAIN FUNCTION

def main(mode):
    banner('STEP 1 - LOAD chase_labels.xlsx')
    labels = pd.read_excel(LABELS_XLS_PATH)
    logger.info(f'{labels.shape[0]} labeled events ({(labels["label"]==1).sum()} positive, {(labels["label"]==0).sum()} negative)')

    banner('STEP 2 - LOAD TRACKS + BUILD PAIRWISE FEATURES')
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME)
    df = pd.read_parquet(parquet_path)
    df = trim_to_calibration(df, calibration_secs, frame_number_end)
    pairs = build_pairs(df, pixels_per_cm)
    pairs['min_alignment_either_deg'] = pairs['min_alignment_either_deg'].fillna(180)  # "no burst = no aim to report" - worst-case value
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s']].head().to_string())

    banner(f'STEP 3 - SLICE SEQUENCES (mode={mode})')
    all_sequences = build_sequences(pairs, labels, mode, RANDOM_SEED, WINDOW_SIZE_FRAMES, STRIDE_FRAMES, MAX_WINDOWS_PER_NEGATIVE_EVENT) # returns  a flat list based on ground_truth chasing file of dicts, each {'event_id': ..., 'label': ..., 'sequence_df': ...}, ready to be split into train/test and fed to the model.
    n_pos = sum(1 for s in all_sequences if s['label'] == 1)
    n_neg = sum(1 for s in all_sequences if s['label'] == 0)
    unit = 'events' if mode == 'whole_event' else 'windows'
    logger.info(f'{len(all_sequences)} total {unit} ({n_pos} positive, {n_neg} negative)')
    split_folder = os.path.join(os.path.dirname(parquet_path), 'chase_train_test')
    train_event_ids = set(pd.read_parquet(os.path.join(split_folder, 'train_df.parquet'))['event_id'].unique()) # which event IDs ended up on the training side of the split, the last time build_chase_windows.py ran?"
    test_event_ids = set(pd.read_parquet(os.path.join(split_folder, 'test_df.parquet'))['event_id'].unique()) # which event IDs ended up on the testing side of the split, the last time build_chase_windows.py ran?"
    train_sequences = [s for s in all_sequences if s['event_id'] in train_event_ids]
    test_sequences = [s for s in all_sequences if s['event_id'] in test_event_ids]
    logger.info(f'{len(train_sequences)} train {unit} ({sum(s["label"] for s in train_sequences)} positive), '
                f'{len(test_sequences)} test {unit} ({sum(s["label"] for s in test_sequences)} positive)')






    banner('STEP 4 - REUSE PHASE D/E TRAIN/TEST SPLIT (same event_ids)')


# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['whole_event', 'windowed'], default='whole_event')
    args = parser.parse_args()
    main(args.mode)
