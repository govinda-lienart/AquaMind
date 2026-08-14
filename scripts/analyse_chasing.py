"""usage e.g python -m scripts.analyse_chasing --video_name IMG_2349"""

#---------------
# IMPORTS
#---------------

import os
import yaml
import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg') # avoids popup windows of poduced plots
import matplotlib.pyplot as plt 
from scripts.console import banner, banner_sub
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

#---------------
# CONFIGS
#---------------

# loading tank related data
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']

#---------------
# HELPER FUNCTIONS 
#---------------

def grab_video_name(video_name):
    "grabs arguments from user and pulls out the related parameters from config.yaml"
    video_cfg = cfg['videos'][video_name]
    parquet_path = video_cfg['parquet_path']
    tank_width_px = video_cfg['tank_width_px']
    tank_width_cm = video_cfg['tank_width_cm']
    calibration_secs = video_cfg['calibration_secs']
    surface_y_px = video_cfg['surface_y_px']
    bottom_y_px = video_cfg['bottom_y_px']
    pixels_per_cm = tank_width_px / tank_width_cm
    banner('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}, surface_y_px = {surface_y_px}, bottom_y_px = {bottom_y_px}')
    return parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px

#---------------
# MAIN FUNCTION
#---------------


def main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px):
    
    banner('COMPUTING')
    df = pd.read_parquet(parquet_path)
    banner_sub('LOADING PARQUET FILE')
    logger.info(f'{df.head().to_string()}')
    banner_sub('EXAMPLE FILTER SELECT ABOVE CALIBRATION')
    print((df['timestamp'] >= calibration_secs).head())
    banner_sub('APPLYING FILTER WITH MASK')
    df = df[df['timestamp'] >= calibration_secs]
    logger.info(f'{df.head().to_string()}')
    banner_sub('DATAFRAME - AFTER CALIBRATION TRIM')
    logger.info(df.shape)
    banner_sub('SELF-MERGE ON FRAME_NUMBER — combine every fish with every other fish, per frame')
    pairs = df.merge(df, on='frame_number', suffixes=('_a', '_b')) #  every row with frame_number == 600 from the left copy gets paired with every row with frame_number == 600 from the right copy, one at a time.
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b']].head(16).to_string())
    banner_sub('PAIRS - SHAPE')
    logger.info(pairs.shape)
    banner_sub('PAIRS - COLUMNS')
    logger.info(pairs.columns) 
    pairs = pairs[pairs['fish_id_a'] < pairs['fish_id_b']] #  Row (1,1) → 1 < 1 → False # Row (1,2) → 1 < 2 → True #Row (2,3) → 2 < 3 → True
    banner_sub('PAIRS - SHAPE AFTER FILTER')
    logger.info(pairs.shape)
    banner_sub('DISTANCE BETWEEN PAIRS')
    pairs['distance_cm'] = np.hypot(pairs['x_a'] - pairs['x_b'], pairs['y_a'] - pairs['y_b']) / pixels_per_cm  # see drawing under script
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm']].head(6).to_string())
    banner_sub('SORTING PAIRS BY PAIR AND FRAME')
    pairs = pairs.sort_values(['fish_id_a', 'fish_id_b', 'frame_number'])
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm']].head(10).to_string())
    banner_sub('CLOSING DISTANCE BY PAIR - OBJECT ')
    grouped_pairs = pairs.groupby(['fish_id_a', 'fish_id_b']) # doesnt compute yet..just spliting the data in groups - one bucket per distinc pair  # object - lazily group by
    banner_sub('DELTA DISTANCE - How much the gap between the two fish changed from the previous frame to this one')
    pairs['delta_distance_cm'] = grouped_pairs ['distance_cm'].diff() # diff => for each row, subtract the value in the row immediately above it (in whatever order the data currently sits in
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'delta_distance_cm']].head(10).to_string())
    pairs['delta_timestamp'] = grouped_pairs ['timestamp_a'].diff()  #  the time elapsed between those same two frames,
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'delta_distance_cm', 'delta_timestamp']].head(10).to_string())
    banner_sub('CLOSING SPEED')
    pairs['closing_speed_cm_s'] = -pairs['delta_distance_cm'] / pairs['delta_timestamp'] # see appendix below - when fish get closer, distance goes from 10 (previous frame) to 8 (this frame) = -2, so delta_distance_cm is negative. Flipping the sign makes closing_speed positive when fish are approaching — more intuitive to read.
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'delta_distance_cm', 'closing_speed_cm_s']].head(10).to_string())
    
    banner('OUTPUT FOLDER')
    output_folder = os.path.dirname(parquet_path) # parquet_path = 'output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet' # dirname() strips the filename off, leaving just the folder:# output_folder = 'output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853'
    figure_dir = os.path.join(output_folder, "output_analyse_chasing")
    os.makedirs(figure_dir, exist_ok=True)













#---------------
# ENTRY POINT
#---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imports video configs")
    parser.add_argument("--video_name",
                        default="IMG_2349",
                        help="indicate which video you want to analyse")
    args = parser.parse_args()
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px)

 
# APPENDIX 

# calculation distance

"""                 fish A (x_a, y_a)
                         *
                         |\
                         | \
              dy = y_a-y_b \   <- straight-line distance
                         |   \     = hypotenuse
                         |    \
                         |     \
                         *------* 
                                 fish B    (x_a, y_b)  <- imaginary corner point
                    (x_b, y_b)
                       dx = x_a - x_b"""


# calculation closing speed
# CLOSING SPEED — worked examples (real ~60fps data, delta_t ~= 0.01667s per frame)
#
# when fish get closer, distance goes from 10 (previous frame) to 8 (this frame) = -2,
# so delta_distance_cm is negative. flipping the sign makes closing_speed positive
# when fish are approaching — more intuitive to read.
#
# 1) SLOW APPROACH
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 199    3.31667    5.00         —                  —         —
# 200    3.33333    4.95         -0.05               0.01667   -(-0.05)/0.01667 = 3.0
#
# a fish closing the gap by just 0.05cm in one frame (sub-millimeter, barely visible)
# scales up to a closing_speed of 3.0 cm/s once expressed "per second" —
# because 0.05cm x 60 frames/sec ~= 3 cm/s
#
# 2) FAST APPROACH / BURST
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 250    4.16667    3.20         —                  —         —
# 251    4.18333    3.05         -0.15               0.01667   -(-0.15)/0.01667 = 9.0
#
# a bigger per-frame drop (0.15cm instead of 0.05cm) scales up to a closing_speed
# of 9.0 cm/s — three times faster than the slow-approach example above
#
# 3) SEPARATING
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 300    5.00000    4.00         —                  —         —
# 301    5.01667    4.10         +0.10               0.01667   -(+0.10)/0.01667 = -6.0
#
# distance grew by 0.10cm this frame, so delta_distance_cm is POSITIVE — flipping
# the sign turns that into a NEGATIVE closing_speed, meaning "moving apart, not
# approaching." this is why the sign convention matters: positive closing_speed
# = closing in (candidate chase), negative closing_speed = separating.