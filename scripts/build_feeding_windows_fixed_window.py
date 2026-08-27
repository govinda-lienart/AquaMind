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

# CONSTANTS

LINK_POS_LABELS = 'output_fish_tracker/feeding_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
WINDOW_SIZE_FRAMES = 45  # the fixed window length you've been using throughout - was calculated based on minumum size of one feeding strike

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
output_folder = os.path.dirname(parquet_path)
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

# STEP 4  — the window slicer function 

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

# STEP 5: BUILD POSITIVE WINDOWS