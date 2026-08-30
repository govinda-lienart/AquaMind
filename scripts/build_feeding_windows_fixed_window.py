"""
usage: python -m scripts.build_feeding_windows_fixed_window

- Loads 118 manually-labeled feeding strikes from feeding_labels.xlsx
- Builds matched fixed-size (45-frame) negative windows from the sand-injection segment
- Summarizes both into one windows_df (speed/burst stats per window). 
- Splits into train/test (stratified by label) 
- Saves both as parquet files in feeding_train_test/.
"""

# IMPORTS

import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name, trim_to_calibration
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

# CONSTANTS

LINK_POS_LABELS = 'output_fish_tracker/feeding_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
WINDOW_SIZE_FRAMES = 45  # the fixed window length you've been using throughout - was calculated based on minumum size of one feeding strike
SAND_START_FRAME = 7505 # prefeeding fase - start control
SAND_END_FRAME = 14340 # prefeeding fase - end control
RANDOM_SEED = 42


# MAIN

#  STEP 1 LOADING / CLEANING DATAFRAME
banner("STEP 1: loading xls file and cleaning dataframe positive labels")

    # load the excel file
pos_labels = pd.read_excel(LINK_POS_LABELS)
logger.info(pos_labels.head().to_string())

    # drop the junk Unnamed columns
banner_sub("drop the junk Unnamed columns") 
pos_labels = pos_labels.drop(columns=["Unnamed: 5","Unnamed: 6"])

    # force the two frame columns start and end to numeric, bad values -> NaN
banner_sub("force the two frame columns start and end to numeric, bad values -> NaN")
pos_labels ["framenumber_start"] = pd.to_numeric(pos_labels["framenumber_start"], errors="coerce")
pos_labels ["framenumber_end"] = pd.to_numeric(pos_labels["framenumber_end"], errors="coerce")

    # drop rows missing any of the three required columns
banner_sub("drop rows missing any of the three required columns")
pos_labels = pos_labels.dropna(subset=["fish_id", "framenumber_start", "framenumber_end"]) # check if any NaN

    # drop the swapped-typo row: end must be >= start
banner_sub("drop the swapped-typo row: end must be >= start")
pos_labels = pos_labels[pos_labels["framenumber_start"] <= pos_labels["framenumber_end"]]
logger.info(pos_labels.head().to_string())

logger.info(f'\n data frame contains {pos_labels.shape[0]} valid labeled rows of strikes across {pos_labels.shape[1]} columns')

#  STEP 2 LOAD TRACKER OUTPUT
banner("STEP 2: LOAD TRACKER OUPUT")
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME )
tracks = pd.read_parquet(parquet_path)
tracks = tracks[tracks['timestamp'] >= calibration_secs]

logger.info(tracks.head().to_string())
logger.info(f'\n data frame contains {tracks.shape[0]} records with {tracks.shape[1]} columns')

#  STEP 3 PER FISH SPEED + BURST
banner("STEP 3: PER FISH SPEED + BURST")
tracks = tracks.sort_values(["fish_id", "frame_number"])          # fish_id first, then frame_number - so diff() stays within one fish's own timeline
grouped_fish = tracks.groupby("fish_id")
    
    # distance
delta_x = grouped_fish['x'].diff()
delta_y = grouped_fish['y'].diff()
distance = np.hypot(delta_x,delta_y)/pixels_per_cm # distance moved per fish
tracks['distance'] = distance
    
    # timestamps
delta_timestamp = grouped_fish['timestamp'].diff()
tracks['delta_timestamp'] = delta_timestamp
tracks['speed_cm_s'] = distance/ delta_timestamp 
logger.info(tracks.head().to_string())

    # speed : remove unusual peaks and smoothen the graph (to deal with swaps/occlusions that might created unwanted peaks)
banner_sub("cap speed_cm_s at 99.9th percentile")
speed_cap = tracks["speed_cm_s"].quantile(0.999) # e.g baseline alone has max=605.83 cm/s with zero stimulus, so the extreme tail can't be real strikes. revising...a normal chasing max 150 cm /c
tracks["speed_cm_s"] = tracks["speed_cm_s"].clip(upper=speed_cap) # clip outliers
tracks["speed_cm_s_smooth"] = grouped_fish["speed_cm_s"].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean()) # moving average - a centered moving average with a 5-frame window-  this goes in each group fish_id...and will slide, and for each value will take 2 before and 2 after..to replace it with an average.
logger.info(tracks[["distance", "delta_timestamp", "speed_cm_s", "speed_cm_s_smooth"]].head().to_string())

    # burst - accelation (change in speed from one frame to another)
banner_sub("burst - acceleration")
tracks["burst"] = grouped_fish["speed_cm_s_smooth"].transform(lambda s: s.diff()) # how much the speed changed from one frame to the next
logger.info(tracks[["distance", "delta_timestamp", "speed_cm_s", "speed_cm_s_smooth", "burst"]].head().to_string())

banner("STEP 4  — the window slicer function ")
def slice_window(fish_id, center_frame): # here with a fiven frame number(center frame) selecting a fixed sized 45 frame window cut from tracks
    window_start = center_frame - WINDOW_SIZE_FRAMES // 2 
    window_end = window_start + WINDOW_SIZE_FRAMES 
    window = tracks[(tracks["fish_id"] == fish_id) &
                    (tracks["frame_number"] >= window_start) &
                    (tracks["frame_number"] < window_end)]
    return window.sort_values("frame_number")

banner_sub("testing function slice_window()")
test_window = slice_window(1, 1000) # the first number is fish_id, the second is central_frame
logger.info(test_window["frame_number"].head().to_string())
logger.info(f'test window shape: {test_window.shape}')

# STEP 6 BUILD POSITIVE WINDOWS
banner("STEP 5: BUILD POSITIVE WINDOWS")
all_windows = []
for row in pos_labels.itertuples(): #Iterating directly over a DataFrame like this loops over its column names (strings like "event_id", "fish_id", etc.), not its rows! You need .itertuples() to loop over actual rows
    center_frame = (row.framenumber_start + row.framenumber_end) / 2
    window = slice_window(row.fish_id, center_frame)
    window = window.copy() #  it creates a completely independent, brand-new DataFrame with its own memory, fully separate from tracks
    window["label"] = 1
    window["event_id"] = row.event_id
    all_windows.append(window)
logger.info(f'in total {len(all_windows)} positive labels processed as windows')
logger.info(f'first window shape: {all_windows[0].shape}')
logger.info(f'first window label: {all_windows[0]["label"].iloc[0]}, event_id: {all_windows[0]["event_id"].iloc[0]}')

# STEP 7: BUILD NEGATIVE WINDOWS
banner("STEP 6: BUILD NEGATIVE WINDOWS")
banner_sub("build candidate negative windows (non-overlapping, sand segment, random fish_id)") # : cheaply collect every possible candidate center point across all fish , just numbers, (fish_id, center_frame) pairs, no actual data slicing yet, so it's fast even if there are hundreds of candidates.
fish_ids = sorted(tracks["fish_id"].unique()) # [1, 2, 3, 4]
candidate_windows= []
for fish_id in fish_ids:
    window_start = SAND_START_FRAME
    while window_start + WINDOW_SIZE_FRAMES <= SAND_END_FRAME: # inner loop will loop for e.g fish_id = 1 and do many loops on in to collect center numbers, and once finished, will go back to outer loop to trigger the next fish_id
        candidate_center = window_start + WINDOW_SIZE_FRAMES // 2
        candidate_windows.append((fish_id, candidate_center))
        window_start += WINDOW_SIZE_FRAMES
logger.info(f'{len(candidate_windows)} candidate negative windows across {len(fish_ids)} fish')

banner_sub("sample negatives to match positive count")
random.seed(RANDOM_SEED)
sampled_negatives = random.sample(candidate_windows, k=min(len(all_windows), len(candidate_windows))) # all windows currently only holds the positive labels
logger.info(f'{len(sampled_negatives)} negatives sampled') # sampled negative candidate_center

banner_sub("build negative windows from sampled candidates")
next_negative_event = int(pos_labels["event_id"].max()+1)
logger.info(f'the event_id for next_negative_event should start from {next_negative_event}') # sampled negative candidate_center
for fish_id, center_frame in sampled_negatives:
    window = slice_window(fish_id, center_frame) # filters tracks
    window = window.copy() # makes independent copies
    window["label"] = 0 
    window["event_id"] = next_negative_event
    next_negative_event += 1
    all_windows.append(window)

logger.info(f'total labels positive and negative is {len(all_windows)}')

# STEP 7:SUMMARIZE WINDOWS INTO ONE DATAFRAME
banner("STEP 7: SUMMARIZE WINDOWS INTO ONE DATAFRAME")
summary_rows = []
for window in all_windows: # for each loop we exract data from each window and convert it into a dictionary , with some data like speed summarized for this window as for example average.
    summary_rows.append({
        "event_id": window["event_id"].iloc[0],
        "label": window["label"].iloc[0],
        "fish_id": window["fish_id"].iloc[0],
        "mean_speed_cm_s": window["speed_cm_s"].mean(),
        "max_speed_cm_s": window["speed_cm_s"].max(),
        "mean_burst": window["burst"].mean(),
        "max_burst": window["burst"].max(),
        "window_frame_start": window["frame_number"].min(),
        "window_frame_end":   window["frame_number"].max(),
        "occluded_frame_count": window["occluded"].sum(),   # frames where the tracker flagged the fish occluded
        "window_row_count": window.shape[0],                # < WINDOW_SIZE_FRAMES => fish had no tracked row at all for some frame
    })
window_df = pd.DataFrame(summary_rows) # converting the list of dictionary into a proper dataframe
logger.info(f"the shape of the window_df is {window_df.shape}")

# drop windows with any occlusion or missing frame - the CNN+LSTM pipeline needs a complete 45-frame
# crop sequence per window, and train_df/test_df must be the single source of truth for which windows
# are valid (so the geometry baseline and the CNN pipeline compare on the exact same window set)
window_df["has_gap"] = (window_df["occluded_frame_count"] > 0) | (window_df["window_row_count"] < WINDOW_SIZE_FRAMES) # | = element-wise OR

before = len(window_df)
window_df = window_df[~window_df["has_gap"]].drop(columns="has_gap").copy()
logger.info(f"dropped {before - len(window_df)} windows with occlusion/gaps, {len(window_df)} remain "
            f"({(window_df['label']==1).sum()} positive, {(window_df['label']==0).sum()} negative)")
logger.info(f"the total number of positive labeled rows is {(window_df['label']==1).sum()} and the total number of negative rows is {(window_df['label']==0).sum()}")
logger.info(window_df.head().to_string())

# STEP 8:STEP 8 TRAIN/TEST SPLIT
banner("STEP 8 TRAIN/TEST SPLIT")
train_df, test_df = train_test_split(
    window_df,
    test_size=0.2,
    stratify=window_df["label"], # windows_df is roughly 50/50 positive/negative (118/118), stratify=window_df["label"] guarantees both train_df and test_df each end up close to that same 50/50 ratio too
    random_state=RANDOM_SEED 
)

logger.info(f"train_df: {train_df.shape[0]} windows of which {(train_df['label']==1).sum()} positive and {(train_df['label']==0).sum()} negative")
logger.info(f"test_df: {test_df.shape[0]} windows of which {(test_df['label']==1).sum()} positive and {(test_df['label']==0).sum()} negative")


# STEP 9: SAVE TRAIN/TEST ON DISK
run_dir = os.path.dirname(parquet_path)
feeding_train_test_path = os.path.join(run_dir, "feeding_train_test")
os.makedirs(feeding_train_test_path, exist_ok=True)

train_parquet_path = os.path.join(feeding_train_test_path, "train_df.parquet")
test_parquet_path = os.path.join(feeding_train_test_path, "test_df.parquet")

train_df.to_parquet(train_parquet_path)
test_df.to_parquet(test_parquet_path)

logger.info(f'saved train_df -> {train_parquet_path}')
logger.info(f'saved test_df -> {test_parquet_path}')






