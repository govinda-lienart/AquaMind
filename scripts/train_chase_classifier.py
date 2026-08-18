"""usage:  python -m scripts.train_chase_classifier
hardcoded path to train_df/test_df built by build_chase_windows.py"""


# IMPORTS

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner
from scripts.chasing_features import grab_video_name

# CONSTANTS

VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'

# MAIN

# STEP 1 - load train_df / test_df saved by build_chase_windows.py
banner('STEP 1 - LOAD train_df / test_df')
parquet_path, *_ = grab_video_name(VIDEO_RUN_NAME) # i don't need to unpack pixels_per_cm, calibration_secs, surface_y_px, so is a throwaway list *_
split_folder = os.path.join(os.path.dirname(parquet_path), 'chase_train_test') #output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet -> output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/chase_train_test
train_df = pd.read_parquet(os.path.join(split_folder, 'train_df.parquet'))
test_df = pd.read_parquet(os.path.join(split_folder, 'test_df.parquet'))
logger.info(f'train_df: {train_df.shape}, test_df: {test_df.shape}')



