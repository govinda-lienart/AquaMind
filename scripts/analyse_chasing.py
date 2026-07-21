# IMPORT

import os
import yaml
import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# no pop up windows when producing results

from scripts.console import banner, banner_sub  # improves layout when printing in the console
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger=logging.getLogger(__name__)

#-------------------
# CONFIG
#-------------------
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']

#-------------------
# HELPER FUNCTION
#-------------------

def grab_video_name(video_name):
    "grabs arguments from user and pulls out the related parameters from config.yaml"
    video_cfg = cfg['videos'][video_name]
    parquet_path = video_cfg['parquet_path']
    tank_width_px = video_cfg['tank_width_px']
    tank_width_cm = video_cfg['tank_width_cm']
    calibration_secs = video_cfg['calibration_secs']
    pixels_per_cm = tank_width_px / tank_width_cm
    banner('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}')
    return parquet_path, pixels_per_cm, calibration_secs

#-------------------
# MAIN
#-------------------

def main(parquet_path, pixels_per_cm, calibration_secs):
    """assessing chasing behaviour"""

    #-------------------
    # LOADING PARQUET
    #-------------------
    banner('LOADING PARQUET DATA')

    #pulls parquet data into dataframe
    df = pd.read_parquet(parquet_path)

    # removing the calibration time - 10 secs from the dataframe
    df = df[df['timestamp'] >= calibration_secs]
    banner_sub('DATAFRAME - HEAD')
    logger.info(df.head().to_string())
    banner_sub('DATAFRAME - INFO')
    logger.info(df.info())

    # output folder for all chasing figures 
    output_folder = os.path.dirname(parquet_path)
    figure_dir = os.path.join(output_folder, "output_chasing_figures")
    os.makedirs(figure_dir, exist_ok=True)


    # ------------------------------------------------------------
    # PAIRWISE DISTANCE 
    # — build one row per fish-pair per frame,then measure the gap between the two fish (cm).
    # - Chasing lives in the gap between two fish, not in one fish.
    # ------------------------------------------------------------

    banner('PAIRWISE DISTANCE CALCULATION BETWEEN DIFFERENT FISH_ID')

    #  self-merge on frame_number: every fish beside every fish (incl. itself) --> because e,g has 4 fish - 16 rows per frame
    pairs = df.merge(df, on='frame_number', suffixes=('_a', '_b'))
    banner_sub('AFTER SELF-MERGE')
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b']].head(16).to_string()) 
        # "which right rows have frame_number 600?" ->  all four of them
        # left fish 1 → matches right fish 1, 2, 3, 4
        # left fish 2 → matches right fish 1, 2, 3, 4
        # etcetera

    # removing the junk from self merge such as 1) comparing fish 1 with fish 1 (paired with itself) or 2)  fish 1 and 2 appears as 2 and 1 row
    pairs = pairs[pairs["fish_id_a"] < pairs["fish_id_b"]]   # keep one copy of each real pair
        # self-pair 1-1: 1 < 1 => False -> dropped 
        # pair 1-2: 1 < 2 -> True → kept 
        # mirror 2-1: 2 < 1 -> False → dropped 
    banner_sub('FILTERING OUT THE JUNK LIKE SELF PAIRED AND DUPLICATES (1,2)/(2,1)')
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b']].head(6).to_string())

    # calculating distance between pairs of fish
    banner_sub('DISTANCE BETWEEN PAIRS OF FISH')
    pairs['distance_cm'] = np.hypot(pairs['x_a'] - pairs['x_b'],
                                pairs['y_a'] - pairs['y_b']) / pixels_per_cm

    # closing speed = how fast the gap between the two fish changes - frame to frame
    banner_sub('CLOSING SPEED')
    pairs['closing_speed'] = pairs.groupby(["fish_id_a", "fish_id_b"])['distance_cm'].diff() # 
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed']].head(20).to_string())
    #        frame_number  fish_id_a  fish_id_b  distance_cm  closing_speed
    # 17           601          1          2     8.648289      -0.029632
    # 33           602          1          2     8.724420       0.076131       --> it grouped bucket fishpair (1,2) and calcualted between frame difference in distance whic is 8.73 - 8.64 = 0.076 coorect




   



#-------------------
# ENTRY POINT/GUARD
#-------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imports Parquet snapshot")
    parser.add_argument("--video_name",
                        help="path to video configuration in config.yaml")
    args = parser.parse_args()
    parquet_path, pixels_per_cm, calibration_secs = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs)
