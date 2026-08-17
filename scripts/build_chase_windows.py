"""usage:  python -m scripts.build_chase_windows
hardcoded path to curated labels chasing"""


# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, trim_to_calibration, build_pairs

# CONSTANTS

WINDOW_SIZE_FRAMES = 35  # fits shortest labeled event (39 frames), >30-frame sustained-speed precedent
STRIDE_FRAMES = 17  # ~50% overlap - more windows per event, but they're correlated, not independent
LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'

# MAIN

# converting excel to labels dataframe
banner_sub('LOADING chase_labels.xlsx as pd')
labels = pd.read_excel(LABELS_XLS_PATH)
logger.info(labels.head().to_string())

# import parquet tracking file
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name('IMG_2349_appearance_2026_08_12_1926')

# load tracks from parquet and building pairwise pairs (distance, and closing speed)
banner('COMPUTING METRICS PAIRS FROM TRACKER IMG_2349_appearance_2026_08_12_1926')
df = pd.read_parquet(parquet_path)
logger.info(df)
df = trim_to_calibration(df, calibration_secs, frame_number_end)
pairs = build_pairs(df, pixels_per_cm)
logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s']].head().to_string())

# sanity check - filtering one single event
event_row = pairs[(pairs['fish_id_a'] == 1) & (pairs['fish_id_b'] == 4) & (pairs['frame_number'] >= 1011) & (pairs['frame_number'] < 1274)]