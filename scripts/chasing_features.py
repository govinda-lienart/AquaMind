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

    banner_sub(f'STEP 2a - TRIM: drop warm-up (<{calibration_secs}s)' + (f' and food-injection phase (>={frame_number_end})' if frame_number_end is not None else ''))
    df = df[df['timestamp'] >= calibration_secs]
    if frame_number_end is not None:
        df = df[df['frame_number'] < frame_number_end]
    logger.info(f'{df.shape[0]} frames remain after trim')

    return df

def build_pairs(df, pixels_per_cm):
    "self-merges df into every fish-pair-per-frame, then computes distance_cm + closing_speed_cm_s (raw/w5/w15)"

    banner_sub('STEP 2b - SELF-MERGE: pair every fish with every other fish, per frame')
    pairs = df.merge(df, on='frame_number', suffixes=('_a', '_b')) #  every row with frame_number == 600 from the left copy gets paired with every row with frame_number == 600 from the right copy, one at a time.
    pairs = pairs[pairs['fish_id_a'] < pairs['fish_id_b']] #  Row (1,1) → 1 < 1 → False # Row (1,2) → 1 < 2 → True #Row (2,3) → 2 < 3 → True
    logger.info(f'{pairs.shape[0]} pair-frames, {pairs[["fish_id_a", "fish_id_b"]].drop_duplicates().shape[0]} unique pairs')

    banner_sub('STEP 2c - DISTANCE: np.hypot(dx, dy) / pixels_per_cm (see drawing under analyse_chasing.py)')
    pairs['distance_cm'] = np.hypot(pairs['x_a'] - pairs['x_b'], pairs['y_a'] - pairs['y_b']) / pixels_per_cm
    pairs = pairs.sort_values(['fish_id_a', 'fish_id_b', 'frame_number'])

    banner_sub('STEP 2d - SMOOTHING: rolling mean of distance_cm (reduces tracker jitter before differencing)')
    grouped_pairs = pairs.groupby(['fish_id_a', 'fish_id_b']) # object - lazily groups rows into one bucket per distinct pair, nothing computed yet
    pairs['distance_cm_smooth'] = grouped_pairs['distance_cm'].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    # wider window, kept ONLY to compare noise levels against the window=5 version above —
    # not yet decided which one becomes the "real" distance_cm_smooth going forward
    pairs['distance_cm_smooth_w15'] = grouped_pairs['distance_cm'].transform(lambda s: s.rolling(15, min_periods=1, center=True).mean())

    banner_sub('STEP 2e - CLOSING SPEED: how fast the gap is shrinking (+ closing in, - separating)')
    # raw (no smoothing) / w5 / w15 all kept side by side ONLY so the comparison plot in
    # analyse_chasing.py can show the full before/after noise-reduction story
    pairs['delta_distance_cm_raw'] = grouped_pairs['distance_cm'].diff()
    pairs['delta_distance_cm'] = grouped_pairs['distance_cm_smooth'].diff()
    pairs['delta_distance_cm_w15'] = grouped_pairs['distance_cm_smooth_w15'].diff()
    pairs['delta_timestamp'] = grouped_pairs['timestamp_a'].diff()  #  time elapsed between those same two frames
    pairs['closing_speed_cm_s_raw'] = -pairs['delta_distance_cm_raw'] / pairs['delta_timestamp']
    pairs['closing_speed_cm_s'] = -pairs['delta_distance_cm'] / pairs['delta_timestamp'] # see appendix in analyse_chasing.py for the sign-convention walkthrough
    pairs['closing_speed_cm_s_w15'] = -pairs['delta_distance_cm_w15'] / pairs['delta_timestamp']
    logger.info(f'{pairs.shape[0]} rows, columns: distance_cm, closing_speed_cm_s (+ raw/w15 variants)')

    return pairs
