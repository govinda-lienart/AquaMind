"usage: "

# IMPORTS

import pandas as pd
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.video_utils import grab_video_name, trim_to_calibration
from scripts.console import banner, banner_sub

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

# Step 2 — Load train_df/test_df


