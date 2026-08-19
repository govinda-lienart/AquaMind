# IMPORTS

import os
import argparse
import random
from datetime import datetime
import yaml
import pandas as pd
from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, trim_to_calibration, build_pairs
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANT
LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'


# MAIN

banner('STEP 1 - LOAD chase_labels.xlsx')
labels = pd.read_excel(LABELS_XLS_PATH)
logger.info(f'{labels.shape[0]} labeled events ({(labels["label"]==1).sum()} positive, {(labels["label"]==0).sum()} negative)')

banner('STEP 2 - LOAD TRACKS + BUILD PAIRWISE FEATURES')
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME)
df = pd.read_parquet(parquet_path)
df = trim_to_calibration(df, calibration_secs, frame_number_end)
pairs = build_pairs(df, pixels_per_cm)
pairs['min_alignment_either_deg'] = pairs['min_alignment_either_deg'].fillna(180)
logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s']].head().to_string())
