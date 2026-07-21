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
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}, surface_y_px = {surface_y_px}, bottom_y_px = {bottom_y_px}')
    return parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px

#-------------------
# MAIN
#-------------------

def main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px):
    """assessing chasing behaviour"""

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

#-------------------
# ENTRY POINT/GUARD
#-------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imports Parquet snapshot")
    parser.add_argument("--video_name",
                        help="path to video configuration in config.yaml")
    args = parser.parse_args()
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs)
