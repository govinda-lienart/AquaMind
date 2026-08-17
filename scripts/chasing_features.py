"""Shared Stage 7 chasing utilities — no __main__, import-only (see analyse_chasing.py / build_chase_windows.py)."""

#---------------
# IMPORTS
#---------------

import yaml
import numpy as np
from scripts.console import banner, banner_sub
import logging
logger = logging.getLogger(__name__)

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
    frame_number_end = video_cfg.get('frame_number_end')  # optional - .get() returns None if not set, instead of KeyError
    pixels_per_cm = tank_width_px / tank_width_cm
    banner('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}, surface_y_px = {surface_y_px}, bottom_y_px = {bottom_y_px}, frame_number_end = {frame_number_end}')
    return parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end

def trim_to_calibration(df, calibration_secs, frame_number_end):
    "drops tracker warm-up frames (before calibration_secs) and, if set, everything at/after frame_number_end"

    banner_sub('APPLYING CALIBRATION TRIM')
    df = df[df['timestamp'] >= calibration_secs]
    logger.info(df.shape)

    if frame_number_end is not None:
        banner_sub(f'CUTTING VIDEO AT frame_number_end = {frame_number_end}')
        df = df[df['frame_number'] < frame_number_end]
        logger.info(df.shape)

    return df

def build_pairs(df, pixels_per_cm):
    "self-merges df into every fish-pair-per-frame, then computes distance_cm + closing_speed_cm_s (raw/w5/w15)"

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
    pairs['distance_cm'] = np.hypot(pairs['x_a'] - pairs['x_b'], pairs['y_a'] - pairs['y_b']) / pixels_per_cm  # see drawing under analyse_chasing.py
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm']].head(6).to_string())
    banner_sub('SORTING PAIRS BY PAIR AND FRAME')
    pairs = pairs.sort_values(['fish_id_a', 'fish_id_b', 'frame_number'])
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm']].head(10).to_string())

    banner_sub('CLOSING DISTANCE BY PAIR - OBJECT ')
    grouped_pairs = pairs.groupby(['fish_id_a', 'fish_id_b']) # doesnt compute yet..just spliting the data in groups - one bucket per distinc pair  # object - lazily group by
    banner_sub('SMOOTHING DISTANCE (rolling average, reduces tracker jitter before differencing)')
    pairs['distance_cm_smooth'] = grouped_pairs['distance_cm'].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    # wider window, kept ONLY to compare noise levels against the window=5 version above —
    # not yet decided which one becomes the "real" distance_cm_smooth going forward
    pairs['distance_cm_smooth_w15'] = grouped_pairs['distance_cm'].transform(lambda s: s.rolling(15, min_periods=1, center=True).mean())

    banner_sub('DELTA DISTANCE - How much the gap between the two fish changed from the previous frame to this one')
    # raw (no smoothing at all) — kept here ONLY so the closing-speed comparison plot can show
    # the full before/after evolution: raw -> window=5 -> window=15
    pairs['delta_distance_cm_raw'] = grouped_pairs['distance_cm'].diff()
    pairs['delta_distance_cm'] = grouped_pairs['distance_cm_smooth'].diff()
    pairs['delta_distance_cm_w15'] = grouped_pairs['distance_cm_smooth_w15'].diff()
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'distance_cm_smooth', 'delta_distance_cm']].head(10).to_string())
    pairs['delta_timestamp'] = grouped_pairs['timestamp_a'].diff()  #  the time elapsed between those same two frames,
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'delta_distance_cm', 'delta_timestamp']].head(10).to_string())

    banner_sub('CLOSING SPEED')
    pairs['closing_speed_cm_s_raw'] = -pairs['delta_distance_cm_raw'] / pairs['delta_timestamp']
    pairs['closing_speed_cm_s'] = -pairs['delta_distance_cm'] / pairs['delta_timestamp'] # see appendix in analyse_chasing.py - when fish get closer, distance goes from 10 (previous frame) to 8 (this frame) = -2, so delta_distance_cm is negative. Flipping the sign makes closing_speed positive when fish are approaching — more intuitive to read.
    pairs['closing_speed_cm_s_w15'] = -pairs['delta_distance_cm_w15'] / pairs['delta_timestamp']
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'delta_distance_cm', 'closing_speed_cm_s', 'closing_speed_cm_s_w15']].head(10).to_string())

    return pairs
