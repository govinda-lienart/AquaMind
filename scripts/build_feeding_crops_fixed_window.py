"""
- Turns each gap-free window into an ordered sequence of 45 crop-image paths.
usage: python -m scripts.build_feeding_crops_fixed_window
"""
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name

VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'

# STEP 1 — load the gap-free window parquets
banner("STEP 1 — load train_df / test_df") 
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME) 
output_folder = os.path.dirname(parquet_path) # output_fish_tracker/tracker_IMG_2349_appearance_2026_08_12_1926
feeding_train_test_path = os.path.join(output_folder, 'feeding_train_test')
df_train_path = os.path.join(feeding_train_test_path, 'train_df.parquet')
df_test_path = os.path.join(feeding_train_test_path, 'test_df.parquet')
df_train = pd.read_parquet(df_train_path)
df_test = pd.read_parquet(df_test_path)
banner_sub("train_df") 
logger.info(df_train.head().to_string())

# STEP 2 — the 45 ordered crop paths for one window

# STEP 2a — take one window (example first row), read its fish_id + frame range 
sample = df_train.iloc[0]    # pull out first row of the dataframe
banner_sub("first row") 
logger.info(sample.to_string())
fish_id = int(sample["fish_id"]) # wrap as integer - easier to use as filename later
window_frame_start = int(sample["window_frame_start"])
window_frame_end = int(sample["window_frame_end"])
logger.info(f"\nfish_id: {fish_id}, window_frame_start: {window_frame_start}, window_frame_end: {window_frame_end}")

# STEP 2b — build the list of 45 file crop paths for the selected window 
crops_path = os.path.join(output_folder, 'crops')
fish_dir = os.path.join(crops_path, f"fish_{fish_id}")

crop_list_paths = []

for frame in range(window_frame_start, window_frame_end+1):
    path = os.path.join(fish_dir,f"frame_{frame}_fish_{fish_id}.jpg")
    crop_list_paths.append(path)

logger.info(f"{len(crop_list_paths)} paths")
logger.info(f"first: {crop_list_paths[0]}")
logger.info(f"last:  {crop_list_paths[-1]}")
logger.info(f"first exists: {os.path.exists(crop_list_paths[0])}")


 