"""usage:  python -m scripts.build_chase_windows
hardcoded path to curated labels chasing"""


# IMPORTS

import random
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
MAX_WINDOWS_PER_NEGATIVE_EVENT = 3  # caps negatives so they don't swamp the (fewer) positive windows
RANDOM_SEED = 42  # fixed seed - same pair gets sampled for each negative event every time this runs

# MAIN

# STEP 1 - load the human-labeled chase/non-chase events
banner('STEP 1 - LOAD chase_labels.xlsx')
labels = pd.read_excel(LABELS_XLS_PATH)
logger.info(f'{labels.shape[0]} labeled events ({(labels["label"]==1).sum()} positive, {(labels["label"]==0).sum()} negative)')

# STEP 2 - load the tracker output for this video and build per-frame pairwise features
banner('STEP 2 - LOAD TRACKS + BUILD PAIRWISE FEATURES (IMG_2349_appearance_2026_08_12_1926)')
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name('IMG_2349_appearance_2026_08_12_1926')
df = pd.read_parquet(parquet_path)
df = trim_to_calibration(df, calibration_secs, frame_number_end)
pairs = build_pairs(df, pixels_per_cm)
logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s']].head().to_string())

# STEP 3 - sanity check: prove the masking + sliding-window slicing works on one real event (event 2)
banner('STEP 3 - SANITY CHECK on one event (event 2, fish 1-4, frames 1011-1274)')
event_two = pairs[(pairs['fish_id_a'] == 1) & (pairs['fish_id_b'] == 4) & (pairs['frame_number'] >= 1011) & (pairs['frame_number'] < 1274)] # selecting event 2
logger.info(event_two.shape) 
event_two_window_1 = event_two[(event_two['frame_number'] >= 1011) & (event_two['frame_number'] < 1011 + 35)] # selecting first window of event 2
logger.info(f'window_1 {event_two_window_1.shape}') 
event_two_window_2 = event_two[(event_two['frame_number'] >= 1011+17) & (event_two['frame_number'] < 1011 + 17 + 35)] # sliding to second window...overlapping of 17 frames
logger.info(f'window_2 {event_two_window_2.shape}') 


# the reusable version of the sanity-check above: any fish pair, any frame range
def slice_windows(fish_id_a, fish_id_b, start_frame, end_frame):
    event_two = pairs[(pairs['fish_id_a'] == fish_id_a) & (pairs['fish_id_b'] == fish_id_b) & (pairs['frame_number'] >= start_frame) & (pairs['frame_number'] < end_frame)]
    windows = [] # is a list of dataframes
    window_start = start_frame
    while window_start + WINDOW_SIZE_FRAMES <= end_frame:
        window = event_two[(event_two['frame_number'] >= window_start) & (event_two['frame_number'] < window_start + WINDOW_SIZE_FRAMES)]
        windows.append(window)
        window_start += STRIDE_FRAMES
    return windows

# STEP 4 - build windows for every POSITIVE labeled event (negatives come later - need pair-sampling first)
banner('STEP 4 - BUILD WINDOWS FOR ALL POSITIVE EVENTS')
positive_labels = labels[labels['label'] == 1] # boolean mask - creating a dataframe with all the positive labels (label = 1)
logger.info(positive_labels.head().to_string())
print()
all_windows = [] # one flat list, each dfN being one 35-frame window's worth of data.

for row in positive_labels.itertuples(): # itertuple is a tuple but allows to grab a value by its name rather than by its position...example row.fish_id
    logger.info(f'row: {row}')
    event_windows = slice_windows(row.fish_id_a, row.fish_id_b, row.framenumber_start, row.framenumber_end) # for the current event (one row), pull its fish IDs and frame range, and feed them into slice_windows() to build that one event's windows.
    all_windows.extend(event_windows)  # extend, not append - flattens each event's window-list into one big list
                                        # my_list is now [1, 2, 3, [4, 5]]   <- the [4, 5] is nested inside as ONE item
                                        # my_list is now [1, 2, 3, 4, 5]   <- 4 and 5 got added separately, flattened in

logger.info(f'\n{len(all_windows)} total positive windows, from {len(positive_labels)} positive events')
logger.info(f'\n printing first df of the windows list: \n {all_windows[0].head(100)}')


# STEP 5 - figure out which fish pairs actually exist, for sampling negative-event pairs next
banner('STEP 5 - AVAILABLE PAIRS (for negative-event sampling)')
unique_pairs = pairs[['fish_id_a', 'fish_id_b']].drop_duplicates()
logger.info(unique_pairs.to_string())

# STEP 6 turning unique_pairs (a dataframe) into a plain list of (fish_id_a, fish_id_b) tuples,
# since random.choice() needs a plain list to pick from, not a dataframe
pair_list = list(unique_pairs.itertuples(index=False, name=None)) # itertuples(...)  through unique_pairs row by row, handing back one tuple-like object per row.
logger.info(f'pair_list: {pair_list}')

# STEP 7 - build windows for every NEGATIVE labeled event (fish_id_a/fish_id_b are NaN for these
# rows - a negative label doesn't say which pair - so sample one pair per event instead)
banner('STEP 7 - BUILD WINDOWS FOR ALL NEGATIVE EVENTS')
random.seed(RANDOM_SEED)
negative_labels = labels[labels['label'] == 0]

for row in negative_labels.itertuples():
    sampled_fish_id_a, sampled_fish_id_b = random.choice(pair_list)
    event_windows = slice_windows(sampled_fish_id_a, sampled_fish_id_b, row.framenumber_start, row.framenumber_end)
    capped_windows = event_windows[:MAX_WINDOWS_PER_NEGATIVE_EVENT]  # only take the first few - avoids negatives outnumbering positives
    logger.info(f'event {row.event_id}: sampled pair ({sampled_fish_id_a}, {sampled_fish_id_b}), {len(event_windows)} possible windows, {len(capped_windows)} kept')
    all_windows.extend(capped_windows) # add to the existing windows list

logger.info(f'\n{len(all_windows)} total windows (positive + negative combined)')







