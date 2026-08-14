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
    
    # reading parquet file and reorganizing df
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



















# APPENDIX 

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

 
