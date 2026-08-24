"""usage:  python -m scripts.build_chase_windows
hardcoded path to curated labels chasing"""


# IMPORTS

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, trim_to_calibration, build_pairs

# CONSTANTS

WINDOW_SIZE_FRAMES = 35  # fits shortest labeled event (39 frames), >30-frame sustained-speed precedent
STRIDE_FRAMES = 17  # ~50% overlap - more windows per event, but they're correlated, not independent
LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'
MAX_WINDOWS_PER_NEGATIVE_EVENT = 4  # caps negatives so they don't swamp the (fewer) positive windows
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
    for window in event_windows: # this inner loop adds 2 additonal columns so i know in the all_windows list which will contain all the windows which one is neg or positve
        window['label'] = 1
        window['event_id'] = row.event_id
        window['chaser_id'] = row.chaser_id  # untrusted metadata (per chase_labels.csv schema notes) but useful for manual review
    
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
    for window in capped_windows:
        window['label'] = 0
        window['event_id'] = row.event_id
        window['chaser_id'] = row.chaser_id  # always NaN for negatives - no chaser
    logger.info(f'event {row.event_id}: sampled pair ({sampled_fish_id_a}, {sampled_fish_id_b}), {len(event_windows)} possible windows, {len(capped_windows)} kept')
    all_windows.extend(capped_windows) # add to the existing windows list


# eyeball one real positive window and one real negative window side by side
display_cols = ['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s', 'label', 'event_id']
one_positive_window = next(w for w in all_windows if w['label'].iloc[0] == 1)
one_negative_window = next(w for w in all_windows if w['label'].iloc[0] == 0)
logger.info(f'\none POSITIVE window (event_id={one_positive_window["event_id"].iloc[0]}):\n{one_positive_window[display_cols].to_string()}') # next(...)  walks through until it finds one match, grabs it, and stops immediately — it never even looks at the rest of the list.
logger.info(f'\none NEGATIVE window (event_id={one_negative_window["event_id"].iloc[0]}):\n{one_negative_window[display_cols].to_string()}') 

# total pos/neg
n_positive_windows = sum(1 for w in all_windows if w['label'].iloc[0] == 1)
n_negative_windows = sum(1 for w in all_windows if w['label'].iloc[0] == 0)
logger.info(f'\n{len(all_windows)} total windows ({n_positive_windows} positive, {n_negative_windows} negative)')

# STEP 8 - sanity check: summarize ONE window (35 raw rows) into fixed columns (a handful of numbers)
banner('STEP 8 - SANITY CHECK: summarize one window into fixed columns')
mean_distance = one_positive_window['distance_cm'].mean()
min_distance = one_positive_window['distance_cm'].min()
max_closing_speed = one_positive_window['closing_speed_cm_s'].max()
logger.info(f'mean_distance={mean_distance}, min_distance={min_distance}, max_closing_speed={max_closing_speed}')

# STEP 9 - generalize STEP 8: summarize EVERY window into one row, building the final dataset for Phase E
banner('STEP 9 - SUMMARIZE ALL WINDOWS INTO ONE DATAFRAME')
summary_rows = []

for window in all_windows:
    summary_rows.append({
        'event_id': window['event_id'].iloc[0],
        'label': window['label'].iloc[0],
        'fish_id_a': window['fish_id_a'].iloc[0],  # already on window (sliced from pairs) - true for positives AND sampled negatives alike
        'fish_id_b': window['fish_id_b'].iloc[0],
        'chaser_id': window['chaser_id'].iloc[0],  # NaN for negatives - no chaser
        'window_frame_start': window['frame_number'].min(),  # this window's own 35-frame slice, NOT the whole event's range
        'window_frame_end': window['frame_number'].max(),
        # every key below is a WINDOW-LEVEL AGGREGATE (one number summarizing all 35 frames) - named
        # with a _summary suffix specifically because several source columns on `pairs`/`window` share
        # the SAME base name (e.g. window['max_burst_either'] is itself already a PER-FRAME order-invariant
        # value) - without the suffix, 'max_burst_either' would mean two different things in two DataFrames
        'mean_distance_cm_summary': window['distance_cm'].mean(),
        'min_distance_cm_summary': window['distance_cm'].min(),
        'max_distance_cm_summary': window['distance_cm'].max(),
        'mean_closing_speed_cm_s_summary': window['closing_speed_cm_s'].mean(),
        'min_closing_speed_cm_s_summary': window['closing_speed_cm_s'].min(),
        'max_closing_speed_cm_s_summary': window['closing_speed_cm_s'].max(),
        'mean_speed_either_cm_s_summary': window['max_speed_either'].mean(),  # order-invariant - whichever fish moved faster each frame
        'max_speed_either_cm_s_summary': window['max_speed_either'].max(),
        'mean_burst_either_summary': window['max_burst_either'].mean(),  # order-invariant - whichever fish accelerated harder each frame
        'max_burst_either_summary': window['max_burst_either'].max(),
        'min_alignment_either_deg_summary': window['min_alignment_either_deg'].min(skipna=True) if window['min_alignment_either_deg'].notna().any() else 180,  # NaN when NEITHER fish burst in this window at all - sentinel 180deg = "no aim to report" (worst possible), since sklearn can't handle NaN
    })

windows_df = pd.DataFrame(summary_rows)  # list of dicts -> one row per dict, keys become columns
logger.info(f'\nwindows_df shape: {windows_df.shape}')
logger.info(windows_df.head(10).to_string())

# STEP 10 - get one row per unique EVENT (not per window) with its label - this is what
# gets split into train/test, never windows_df directly, or windows would leak across the split
banner('STEP 10 - UNIQUE EVENTS (for event-level train/test split)')
events_with_labels = windows_df[['event_id', 'label']].drop_duplicates()
logger.info(f'{events_with_labels.shape[0]} unique events')
logger.info(events_with_labels.to_string())

# STEP 11 - split EVENTS (not windows) into train/test, stratified so both sides get a
# proportional mix of positive/negative events, seeded for reproducibility
banner('STEP 11 - EVENT-LEVEL TRAIN/TEST SPLIT')
train_events, test_events = train_test_split( #sklearn function
    events_with_labels,
    test_size=0.3, # fraction of your data to hold back for testing usually .2 is 20 percent
    stratify=events_with_labels['label'],
    random_state=RANDOM_SEED,
)
logger.info(f'{train_events.shape[0]} train events, {test_events.shape[0]} test events')

# STEP 12 -  split windows_df itself, by whichever side each window's event_id landed on
train_df = windows_df[windows_df['event_id'].isin(train_events['event_id'])]
test_df = windows_df[windows_df['event_id'].isin(test_events['event_id'])]
logger.info(f'train_df: {train_df.shape[0]} windows ({(train_df["label"]==1).sum()} positive, {(train_df["label"]==0).sum()} negative)')
logger.info(f'test_df: {test_df.shape[0]} windows ({(test_df["label"]==1).sum()} positive, {(test_df["label"]==0).sum()} negative)')

# first prodcued with 0.2->  train_df is 84 positive / 61 negative (balanced-ish), while test_df is 5 positive / 16 negative (heavily skewed negative).
# problem has zero knowledge or control over how many windows each event produces
# by changing test size to 0.3 (30% test / 70% train), pulled in more test events (13 vs 9),
# which reduced the chance of an unlucky all-short-events draw - improved test balance
# from severely skewed (5 pos/16 neg) to mildly skewed (16 pos/24 neg), still not exact 1:1

# STEP 13 - save train_df/test_df to disk, next to this video's tracks.parquet, so Phase E
# can load them directly instead of rebuilding this whole pipeline every time
banner('STEP 13 - SAVE train_df / test_df')
output_folder = os.path.join(os.path.dirname(parquet_path), 'chase_train_test')  # dedicated subfolder, same pattern as analyse_chasing.py's output_analyse_chasing/
os.makedirs(output_folder, exist_ok=True)
train_path = os.path.join(output_folder, 'train_df.parquet')
test_path = os.path.join(output_folder, 'test_df.parquet')
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
logger.info(f'saved train_df -> {train_path}')
logger.info(f'saved test_df -> {test_path}')
