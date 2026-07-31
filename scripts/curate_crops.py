
""" - scripts selected the crops that are within a clean strech with no swaps 
    - takes an argparse argument dir_outpu
"""

# IMPORTS

import pandas as pd
import numpy as np
import logging
import glob # search file names/paths
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)
import re
import os
from scripts.console import banner, banner_sub
import shutil # to copy files
import argparse

# CONFIG
STEP = 15 # keep every 15th crop when subsampling for diversity
MIN_SEPARATION_PX = 150 # if a fish's nearest neighbour is closer than this many px, its crop may be overlap-contaminated -> drop it (tune this)

# HELPER FUNCTIONS

def load_windows(run_dir):
    """Load the curated-windows CSV and return only the rows for this video's tracker run."""
    banner("LOAD VERIFIED WINDOWS")   # STAGE 1

    # load all the streches (windows) containing frames with no swapping
    windows_csv_path = "output_fish_tracker/curation_windows.csv"
    windows = pd.read_csv(windows_csv_path,sep=";") # ; because csv european
    logger.info(windows.head().to_string())
    logger.info(windows.shape)

    # keep only THIS video's windows (folder name = video_name in the csv)
    banner_sub("filter windows to this video")
    folder_tracked_video = os.path.basename(run_dir) # run_dir = "output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202" - >tracker_IMG_1839_basic_2026_07_23_1202
    my_windows = windows[windows.video_name == folder_tracked_video]  # boolean mask - seleting only output video of interest from the df ----> my_windows = windows[windows["video_name"] == folder_tracked_video]
    return my_windows

def build_crops_table(run_dir):
    """Glob every harvested crop, parse frame_number + fish_id from each filename, return a crops DataFrame."""
    banner("BUILD CROPS TABLE")   # STAGE 2

    # gather every crop path the tracker wrote for this run
    banner_sub("gather crop paths")
    crop_pattern = f"{run_dir}/crops/fish_*/*.jpg" # the * matches all files fish_1, fish_2,.... and same for file jpg...* matches all the images....frame001_fish1.jpg - inert string - nothing happens yet
    crop_paths = glob.glob(crop_pattern) # read crop-pattern string and converts it into a list of paths - glob.gob (fist glob is module, seocnd glob is fucntion - from gob toolbox use glob tool) # it takes path...and creates a list
    logger.info(f"total number of crops detected is {len(crop_paths)}")

    # parse frame_number + fish_id out of each filename into one row
    banner_sub("parse frame + fish id from filenames")
    rows = [] # list of dictionaries
    for path in crop_paths:
        m = re.search(r"frame_(\d+)_fish_(\d+)",path)
        if m is None:                                  # guard - a filename that doesn't match is a bug, fail loud
            raise ValueError(f"filename didn't match expected pattern: {path}")
        frame_num = int(m.group(1)) # the first capture - 298 # frame 298
        fish_id = int(m.group(2)) # second capture - 4 # fish 4
        rows.append({"path":path, "frame_number":frame_num, "fish_id": fish_id}) # dictionary
    # converting the dictionary above into a dataframe -> each dict is one row....each key is a column name
    crops = pd.DataFrame(rows)
    logger.info(crops.head().to_string())
    logger.info(crops.shape)
    return crops

def filter_separated(crops, run_dir, min_dist):
    """Drop crops where the fish's nearest neighbour is closer than min_dist px.
       When two boxes overlap, the crop captures the WRONG fish even though the tracker's ID is correct —
       so a 'no swap' window can still hold contaminated crops. Geometry (pairwise distance) catches them."""
    banner("FILTER OVERLAP-CONTAMINATED CROPS")   # STAGE 2.5

    # positions x,y per fish per frame, from THIS run's tracks (same run that made the crops)
    tracks_path = os.path.join(run_dir, "tracks.parquet")
    tracks = pd.read_parquet(tracks_path)
    logger.info(f"loaded {len(tracks)} track rows from {tracks_path}")

    # for each frame, flag every fish whose closest neighbour is nearer than min_dist
    banner_sub("flag (frame, fish) where nearest neighbour is too close")
    contaminated = set()   # {(frame_number, fish_id), ...}
    for frame_num, g in tracks.groupby("frame_number"):
        xy  = g[["x", "y"]].to_numpy()                   # positions of all fish in this frame
        ids = g["fish_id"].to_numpy()
        if len(ids) < 2:
            continue                                     # a lone fish can't be contaminated by a neighbour
        diff = xy[:, None, :] - xy[None, :, :]           # (n, n, 2) pairwise coordinate differences
        dist = np.hypot(diff[..., 0], diff[..., 1])      # (n, n) pairwise distances (same np.hypot as speed)
        np.fill_diagonal(dist, np.inf)                   # ignore each fish's distance to itself (0)
        nearest = dist.min(axis=1)                       # each fish's closest-neighbour distance
        for fid, d in zip(ids, nearest):
            if d < min_dist:
                contaminated.add((int(frame_num), int(fid)))
    logger.info(f"flagged {len(contaminated)} contaminated (frame, fish) pairs at min_dist={min_dist}px")

    # drop the contaminated crops
    before = len(crops)
    keys = list(zip(crops.frame_number, crops.fish_id))
    kept = crops[[k not in contaminated for k in keys]]
    banner_sub(f"kept {len(kept)} of {before} crops after separation filter")
    return kept

def tag_and_subsample(crops, my_windows, step):
    """Range-join each crop to its curated window, drop out-of-window crops, then we keep every `step`-th crop per (stretch, fish) for diversity."""
    banner("TAG & SUBSAMPLE CROPS")   # STAGE 3

    # tag each crop with the curated window it falls inside (range-join)
    banner_sub("tag crops to their curated window")
    crops["stretch"] = -1 # just adding a column where all values -1 - guitly until proven innocent....-1 not in any window
    for i, w in my_windows.iterrows(): # hands in  one window row at a time so the first iteration has i = 0 with and w = the window row content, so path, fish_id, frame numeber and stored as series ---so e.g  w.frame_start=300
        crop_inside_window = (crops.frame_number >= w.frame_start) & (crops.frame_number <= w.frame_end) # checking if crop frame is inside the curated window
        crops.loc[crop_inside_window, "stretch"] = i #  df.loc[ WHICH_ROWS , WHICH_COLUMNS # here each crop gets assigned to which window in the video it belongs with the value i from the loop
    kept = crops[crops.stretch != -1] # filtering out the crops outside the window
    banner_sub(f"kept {len(kept)} of {len(crops)} crops inside verified windows")
    logger.info(kept.groupby("stretch").size().to_string())

    # subsample every step-th crop per (stretch, fish) so views are diverse, not near-duplicate frames
    banner_sub("subsample every Nth crop per stretch+fish for diversity")
    curated = (kept.sort_values("frame_number") # Sort the whole table so frames go in order: 298, 299, 3
                    .groupby(["stretch", "fish_id"], group_keys=False) # 10 streches x 5 fish (ids 1 to 5) - 50 combnination
                    .apply(lambda g: g.iloc[::step])) # apply loops of the 50 grouped buckets - glues every function run together.... # [start : stop : step] # step is 15 so take evey 15 crop from each window
    banner_sub(f"subsampled to {len(curated)} of {len(kept)} crops")
    logger.info(curated.drop(columns='path').head(4).to_string())                                   # first few rows
    logger.info(curated.groupby(["stretch", "fish_id"]).size().to_string())
    return curated

def copy_out(run_dir, curated):
    """Function copies each curated crop into curated_crops/stretchNN_fishX/ (one folder per stretch+fish)."""
    banner("COPY CURATED CROPS OUT")   # STAGE 4

    curated_root = os.path.join(run_dir, "curated_crops") # # output_fish_tracker/<run>/curated_crops
    if os.path.exists(curated_root):
        shutil.rmtree(curated_root)   # rebuild from scratch so old (now-filtered-out) crops don't linger from a prior run
        logger.info(f"cleared existing {curated_root}")
    logger.info(f"copying into {curated_root}")
    copied = 0
    for i, row in curated.iterrows():
        dest_folder = os.path.join(curated_root, f"stretch{row.stretch:02d}_fish{row.fish_id}") #stretch00_fish1/ with 02d d fromat decimal integer, 2...at least two charcaters...0 pad empty space leading zeros...05..
        os.makedirs(dest_folder, exist_ok=True) # ensuring folder exist...if not then create.
        shutil.copy(row.path, dest_folder) # copy using the path all the files in the destination folder
        copied += 1
    banner_sub(f"copied {copied} of {len(curated)} crops into {curated_root}")

# MAIN FUNCTION

def main(run_dir): 

    my_windows = load_windows(run_dir)  # load the verified-windows CSV and filter to this video's windows
    crops = build_crops_table(run_dir) # loading/building crops df
    crops = filter_separated(crops, run_dir, MIN_SEPARATION_PX)  # NEW: drop overlap-contaminated crops (geometry filter)
    curated = tag_and_subsample(crops, my_windows, STEP)     # interaction between 2 dataframes (windows x crops) using  range join - tagging the crops to which curated window they belong
    copy_out(run_dir, curated)     # saving on disk the selected crops

# ENTRY POINT
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copying the clean crops from curated streches')
    parser.add_argument("--output-dir", 
                            help="folder where crops will be saved e.g output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202")
    args = parser.parse_args()
    main(args.output_dir)
