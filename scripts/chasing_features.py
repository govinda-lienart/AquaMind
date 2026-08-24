"""Shared Stage 7 chasing utilities — no __main__, import-only (see analyse_chasing.py / build_chase_windows.py)."""

#---------------
# IMPORTS
#---------------

import random
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

BURST_THRESHOLD_CM_S = 2.0  # ~95th percentile of |burst| in IMG_2349 - only trust heading/alignment at the moment a fish is actually accelerating, not during ordinary swimming

MIN_SPEED_FOR_HEADING_CM_S = 10  # below this, frame-to-frame direction is mostly tracker-jitter noise, not real movement (same speed-floor fix the TCPA chase-candidate detector used)

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

    banner_sub('STEP 2a2 - INDIVIDUAL SPEED: per-fish speed, computed BEFORE the merge so it becomes speed_cm_s_a/speed_cm_s_b after')
    df = df.sort_values(['fish_id', 'frame_number'])
    grouped_fish = df.groupby('fish_id')
    delta_x = grouped_fish['x'].diff()
    delta_y = grouped_fish['y'].diff()
    delta_distance_cm = np.hypot(delta_x, delta_y) / pixels_per_cm
    delta_timestamp = grouped_fish['timestamp'].diff()
    df['speed_cm_s'] = delta_distance_cm / delta_timestamp

    banner_sub('STEP 2a3 - HEADING: which direction each fish is currently moving (reuses delta_x/delta_y from speed above). Raw for now - angles can\'t be rolling-averaged the naive way (359 and 1 degrees are neighbors, not far apart), so no smoothing fix like speed/distance got yet')
    df['heading_deg'] = np.degrees(np.arctan2(delta_y, delta_x))

    banner_sub('STEP 2b - SELF-MERGE: pair every fish with every other fish, per frame')
    pairs = df.merge(df, on='frame_number', suffixes=('_a', '_b')) #  every row with frame_number == 600 from the left copy gets paired with every row with frame_number == 600 from the right copy, one at a time.
    pairs = pairs[pairs['fish_id_a'] < pairs['fish_id_b']] #  Row (1,1) → 1 < 1 → False # Row (1,2) → 1 < 2 → True #Row (2,3) → 2 < 3 → True
    logger.info(f'{pairs.shape[0]} pair-frames, {pairs[["fish_id_a", "fish_id_b"]].drop_duplicates().shape[0]} unique pairs')

    banner_sub('STEP 2c - DISTANCE: np.hypot(dx, dy) / pixels_per_cm (see drawing under analyse_chasing.py)')
    pairs['distance_cm'] = np.hypot(pairs['x_a'] - pairs['x_b'], pairs['y_a'] - pairs['y_b']) / pixels_per_cm
    pairs = pairs.sort_values(['fish_id_a', 'fish_id_b', 'frame_number'])

    banner_sub('STEP 2c2 - BEARING + ALIGNMENT: is fish A\'s heading pointed at fish B ("aiming a gun"), and vice versa - same frame, no time involved')
    bearing_a_to_b_deg = np.degrees(np.arctan2(pairs['y_b'] - pairs['y_a'], pairs['x_b'] - pairs['x_a']))
    bearing_b_to_a_deg = np.degrees(np.arctan2(pairs['y_a'] - pairs['y_b'], pairs['x_a'] - pairs['x_b']))
    # smallest angular difference, handling the 359°/1° wraparound: shift into (-180, 180] before taking abs
    pairs['alignment_a_deg'] = ((pairs['heading_deg_a'] - bearing_a_to_b_deg + 180) % 360 - 180).abs()  # 0 = fish A aimed straight at B, 180 = aimed straight away
    pairs['alignment_b_deg'] = ((pairs['heading_deg_b'] - bearing_b_to_a_deg + 180) % 360 - 180).abs()
    # min_alignment_either_deg (order-invariant) built further down, AFTER burst - alignment only trusted at the moment of a burst

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

    banner_sub('STEP 2f - SMOOTH SPEED: same rolling-mean fix as distance_cm_smooth, applied to speed_cm_s before differencing into burst')
    pairs['speed_cm_s_a_smooth'] = grouped_pairs['speed_cm_s_a'].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    pairs['speed_cm_s_b_smooth'] = grouped_pairs['speed_cm_s_b'].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())

    banner_sub('STEP 2g - BURST: each fish\'s own acceleration (diff of its own SMOOTHED speed_cm_s, to avoid tracker-jitter spikes closing_speed_cm_s_raw had before smoothing fixed it)')
    pairs['burst_a'] = grouped_pairs['speed_cm_s_a_smooth'].diff()
    pairs['burst_b'] = grouped_pairs['speed_cm_s_b_smooth'].diff()

    banner_sub('STEP 2h - ORDER-INVARIANT: fish_id_a/b is just "lower ID first", not attacker/victim - so take whichever of the pair burst/moved hardest, not a fixed side')
    pairs['max_speed_either'] = pairs[['speed_cm_s_a_smooth', 'speed_cm_s_b_smooth']].max(axis=1)
    pairs['max_burst_either'] = pairs[['burst_a', 'burst_b']].max(axis=1)

    banner_sub(f'STEP 2i - BURST-GATED ALIGNMENT: only trust a fish\'s heading/alignment on frames where IT is bursting (burst >= {BURST_THRESHOLD_CM_S} cm/s) - "aiming" only means something at the moment of commitment, not during ordinary swimming')
    pairs['alignment_a_deg_gated'] = pairs['alignment_a_deg'].where(pairs['burst_a'] >= BURST_THRESHOLD_CM_S)  # .where() keeps the value if True, else NaN
    pairs['alignment_b_deg_gated'] = pairs['alignment_b_deg'].where(pairs['burst_b'] >= BURST_THRESHOLD_CM_S)
    pairs['min_alignment_either_deg'] = pairs[['alignment_a_deg_gated', 'alignment_b_deg_gated']].min(axis=1)  # NaN on frames where neither fish bursts - ignored by pandas .min() later

    return pairs

def slice_event(pairs, fish_id_a, fish_id_b, start_frame, end_frame):
    "returns the WHOLE labelled event as one variable-length sequence (wrapped in a list of 1, so callers can treat it the same as slice_windows' output)"
    event_df = pairs[(pairs['fish_id_a'] == fish_id_a) & (pairs['fish_id_b'] == fish_id_b)
                      & (pairs['frame_number'] >= start_frame) & (pairs['frame_number'] < end_frame)]
    return [event_df.sort_values('frame_number')]

def slice_windows(pairs, fish_id_a, fish_id_b, start_frame, end_frame, window_size_frames, stride_frames):
    "slides a fixed-size window (window_size_frames, every stride_frames) across one labelled event's frame range"
    event_df = pairs[(pairs['fish_id_a'] == fish_id_a) & (pairs['fish_id_b'] == fish_id_b)
                      & (pairs['frame_number'] >= start_frame) & (pairs['frame_number'] < end_frame)]
    windows = []
    window_start = start_frame
    while window_start + window_size_frames <= end_frame:
        window = event_df[(event_df['frame_number'] >= window_start) & (event_df['frame_number'] < window_start + window_size_frames)]
        windows.append(window.sort_values('frame_number'))
        window_start += stride_frames
    return windows

def build_sequences(pairs, labels, mode, random_seed, window_size_frames=None, stride_frames=None, max_windows_per_negative_event=None):
    "for every labelled row: pick a fish pair (given for positives, randomly sampled for negatives), slice per mode ('whole_event' or 'windowed'), return a flat list of {event_id, label, sequence_df} dicts"
    unique_pairs = pairs[['fish_id_a', 'fish_id_b']].drop_duplicates()
    pair_list = list(unique_pairs.itertuples(index=False, name=None))
    random.seed(random_seed)

    all_sequences = []
    for row in labels.itertuples():
        if row.label == 1:
            fish_id_a, fish_id_b = row.fish_id_a, row.fish_id_b
        else:
            fish_id_a, fish_id_b = random.choice(pair_list)

        if mode == 'whole_event':
            sequences = slice_event(pairs, fish_id_a, fish_id_b, row.framenumber_start, row.framenumber_end)
        else:
            sequences = slice_windows(pairs, fish_id_a, fish_id_b, row.framenumber_start, row.framenumber_end, window_size_frames, stride_frames)
            if row.label == 0:
                sequences = sequences[:max_windows_per_negative_event]

        for sequence_df in sequences:
            all_sequences.append({'event_id': row.event_id, 'label': row.label, 'sequence_df': sequence_df})

    return all_sequences
