"""usage"""




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
    pixels_per_cm = tank_width_px / tank_width_cm
    banner('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}')
    return parquet_path, pixels_per_cm, calibration_secs

#---------------
# MAIN FUNCTION
#---------------


def main (parquet_path, pixels_per_cm, calibration_secs):
    df = pd.read_parquet(parquet_path)
    banner_sub ('LOADING PARQUET FILE')
    logger.info (f'quick view on parquet file': {df.to_string()})


#---------------
# ENTRY POINT
#---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imports video configs")
    parser.add_argument("--video_name",
                        help="indicate which video you want to analyse")
    args = parser.parse_arg()
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs)

 
