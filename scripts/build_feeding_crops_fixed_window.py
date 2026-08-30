"""
build_feeding_crops_fixed_window.py — Stage 7 Part 2, Phase C.

- Loads the gap-free window parquets (train_df / test_df)
- Explodes each window row into 45 per-frame rows, each carrying the path to that frame's crop
  (tracker.py already saved the crop images; this only builds the manifest)
- Saves the per-frame manifest (train_crops.parquet / test_crops.parquet) so the embeddings step
  can load each crop and regroup sequences by event_id

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
train_df_path = os.path.join(feeding_train_test_path, 'train_df.parquet')
test_df_path = os.path.join(feeding_train_test_path, 'test_df.parquet')
train_df = pd.read_parquet(train_df_path)
test_df = pd.read_parquet(test_df_path)
banner_sub("train_df") 
logger.info(train_df.head().to_string())

banner("STEP 2 — the 45 ordered crop paths for one window")

banner_sub("STEP 2a — take one window (example first row), read its fish_id + frame range")
sample = train_df.iloc[0]    # pull out first row of the dataframe
banner_sub("first row") 
logger.info(sample.to_string())
fish_id = int(sample["fish_id"]) # wrap as integer - easier to use as filename later
window_frame_start = int(sample["window_frame_start"])
window_frame_end = int(sample["window_frame_end"])
logger.info(f"\nfish_id: {fish_id}, window_frame_start: {window_frame_start}, window_frame_end: {window_frame_end}")

banner_sub("STEP 2b — build the list of 45 file crop paths for the selected window")
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

banner_sub("STEP 2c — wrap it into a function - list of 45 file crop paths for the selected window")
def crop_paths_for_window(fish_id, window_frame_start, window_frame_end):
    fish_id = int(fish_id) # in parquet is strored as a float
    crops_path = os.path.join(output_folder, 'crops')
    fish_dir = os.path.join(crops_path, f"fish_{fish_id}")

    crop_list_paths = []
    for frame in range(int(window_frame_start), int(window_frame_end) + 1):
        path = os.path.join(fish_dir,f"frame_{frame}_fish_{fish_id}.jpg")
        crop_list_paths.append(path)
    return crop_list_paths 

 
banner("STEP 3 — explode the window table into a frame table (45 frames per window)")

banner_sub("STEP 3a — FLAT version (learning reference): train_df only")
rows = []
missing = []

for w in train_df.itertuples(): # for each row in the dataframe train (outer loop)
    paths = crop_paths_for_window(w.fish_id, w.window_frame_start, w.window_frame_end) # capture the list of crops (strings)
    for frame_position, crop_path in enumerate(paths): # inner loops -  EACH OF THE 45 window path within one row - building a dictinary that contains all data (label, fish_id) for each path
             # iteration 1:  i=0, p="a.jpg"
             # iteration 2:  i=1, p="b.jpg"
        if not os.path.exists(crop_path):
            missing.append(crop_path)
        rows.append({
            "event_id": w.event_id,
            "label": w.label,
            "fish_id": w.fish_id,
            "frame_position": frame_position,
            "frame_number": int(w.window_frame_start) + frame_position,
            "crop_path": crop_path,
        })

train_crops = pd.DataFrame(rows)
logger.info(f"train: {len(train_crops)} rows from {len(train_df)} windows, {len(missing)} missing")
logger.info(f"\n{train_crops.head().to_string()}")

banner_sub("STEP 3b — wrapped into a function (runs on both train_df and test_df)")

def build_crop_sequences(windows_df, split_name):

    rows = []
    missing = []

    for w in windows_df.itertuples(): # for each row in the windows dataframe (outer loop)
        paths = crop_paths_for_window(w.fish_id, w.window_frame_start, w.window_frame_end) # capture the list of crops (strings)
        for frame_position, crop_path in enumerate(paths): # inner loops -  EACH OF THE 45 window path within one row - building a dictinary that contains all data (label, fish_id) for each path
                # iteration 1:  i=0, p="a.jpg"
                # iteration 2:  i=1, p="b.jpg"
            if not os.path.exists(crop_path):
                missing.append(crop_path)
            rows.append({
                "event_id": w.event_id,
                "label": w.label,
                "fish_id": w.fish_id,
                "frame_position": frame_position,
                "frame_number": int(w.window_frame_start) + frame_position,
                "crop_path": crop_path,
            })

    crops_df = pd.DataFrame(rows)
    logger.info(f"{split_name}: {len(crops_df)} rows from {len(windows_df)} windows, {len(missing)} missing")
    return crops_df

banner("STEP 3b — build crop sequences for train and test")
train_crops = build_crop_sequences(train_df, "train")
test_crops  = build_crop_sequences(test_df, "test")

banner("STEP 4 — save crop-sequence manifests")

crop_sequences = os.path.join(feeding_train_test_path, 'crop_sequences')
os.makedirs(crop_sequences,  exist_ok=True)
train_crops_path = os.path.join(crop_sequences, 'train_crops.parquet')
test_crops_path = os.path.join(crop_sequences, 'test_crops.parquet')
train_crops.to_parquet(train_crops_path, index=False)
test_crops.to_parquet(test_crops_path, index=False)

logger.info(f"saved -> {train_crops_path}")
logger.info(f"saved -> {test_crops_path}")


