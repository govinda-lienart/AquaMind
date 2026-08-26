# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name, trim_to_calibration


# CONSTANTS

LINK_POS_LABELS = 'output_fish_tracker/feeding_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'

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

#  STEP 3 LOAD TRACKER OUTPUT
