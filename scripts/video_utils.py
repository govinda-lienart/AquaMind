"""Shared video/tracker-output utilities — no __main__, import-only.

Used by both the chasing pipeline (chasing_features.py) and the feeding-strike
pipeline (build_feeding_windows_fixed_window.py etc) - not chasing-specific,
just config.yaml lookups + a calibration trim, so it lives outside chasing_features.py.
"""

#---------------
# IMPORTS
#---------------

import yaml
from scripts.console import banner, banner_sub
import logging
logger = logging.getLogger(__name__)

#---------------
# CONFIGS
#---------------

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
    frame_number_end = video_cfg.get('frame_number_end')  # optional - .get() returns None if not set, instead of KeyError
    pixels_per_cm = tank_width_px / tank_width_cm
    banner_sub('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}, surface_y_px = {surface_y_px}, bottom_y_px = {bottom_y_px}, frame_number_end = {frame_number_end}')
    return parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end

def trim_to_calibration(df, calibration_secs, frame_number_end):
    "drops tracker warm-up frames (before calibration_secs) and, if set, everything at/after frame_number_end"

    banner_sub(f'STEP 2a - TRIM: drop warm-up (<{calibration_secs}s)' + (f' and food-injection phase (>={frame_number_end})' if frame_number_end is not None else ''))
    df = df[df['timestamp'] >= calibration_secs]
    if frame_number_end is not None:
        df = df[df['frame_number'] < frame_number_end]
    logger.info(f'{df.shape[0]} frames remain after trim')

    return df
