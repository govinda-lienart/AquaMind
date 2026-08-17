"""usage:  python -m scripts.build_chase_windows
hardcoded path to curated labels chasing"""

# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, build_pairs

# MAIN

# converting excel to labels dataframe
banner_sub('LOADING chase_labels.xlsx as pd')
labels_xls_path = 'output_fish_tracker/chase_labels.xlsx'
labels = pd.read_excel(labels_xls_path)
logger.info(labels.head().to_string())

# import parquet tracking file
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name('IMG_2349_appearance_2026_08_12_1926')

# load tracks from parquet and building pairwise pairs (distance, and closing speed)

banner_sub('Pairs in tracker file of IMG_2349_appearance_2026_08_12_1926')
df = pd.read_parquet(parquet_path)
logger.info(df)
pairs = build_pairs(df, pixels_per_cm)
logger.info(pairs.head().to_string())
