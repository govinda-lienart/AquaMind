# IMPORTS

import pandas as pd
import logging
import glob # search file names/paths
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)
import re
import os
from scripts.console import banner, banner_sub
import shutil # to copy files

# LOAD

banner("LOAD VERIFIED WINDOWS")
# load with all curated frames no swapping"
windows_csv_path = "output_fish_tracker/curation_windows.csv"
windows = pd.read_csv(windows_csv_path,sep=";")
logger.info(windows.head().to_string())
logger.info(windows.shape)

# extracting the crops

banner("GATHER CROPS")
run_dir = "output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202"
crop_pattern = f"{run_dir}/crops/fish_*/*.jpg" # the * matches all files fish_1, fish_2,.... and same for file jpg...* matches all the images....frame001_fish1.jpg - inert string - nothing happens yet
crop_paths = glob.glob(crop_pattern) # read crop-pattern string and converts it into a list of paths - glob.gob (fist glob is module, seocnd glob is fucntion - from gob toolbox use glob tool) # it takes path...and creates a list
logger.info(f"total number of crops detected is {len(crop_paths)}")

# testing: extract the frame number frome one filename (test) - used regex
first_frame = crop_paths[0] # selecting the very first frame from the list of tthe path crops
m = re.search(r"frame_(\d+)_fish",first_frame ) # re.search(PATTERN, TEXT) here r [raw string so \ is not special or next line] pattern -> anchor 1 frame [text], (\d_) [capture one or more digits, the () allows me to capture what is inside] , anchor 2 fish [text]
frame_num = int(m.group(1)) # note that m.group(0) would give me frame_298_fish_1 while m.group(1) what is inside the () e.g 298 
logger.info(f"for path {first_frame} the extracted frame number is {frame_num}")

# extract frame + fish_id from all crops - make table
banner("BUILD CROPS TABLE")
rows = [] # list of dictionaries
for path in crop_paths:
    m = re.search(r"frame_(\d+)_fish_(\d+)",path)
    frame_num = int(m.group(1)) # the first capture - 298 # frame 298
    fish_id = int(m.group(2)) # second capture - 4 # fish 4
    rows.append({"path":path, "frame_number":frame_num, "fish_id": fish_id}) # dictionary

# converting with pandas rows into in df -? each dict is one row....each key is a column name
crops = pd.DataFrame(rows)
logger.info(crops.head().to_string())
logger.info(crops.shape)

# making sure that video_name from my csv file with curated windows matches the directory of its related  crops
folder_tracked_video = os.path.basename(run_dir) # run_dir = "output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202" - >tracker_IMG_1839_basic_2026_07_23_1202
my_windows = windows[windows.video_name == folder_tracked_video]  # boolean mask - seleting only output video of interest from the df ----> my_windows = windows[windows["video_name"] == folder_tracked_video]  
logger.info(crops.head().to_string())
logger.info(crops.shape)

# tagging the crops to which curated window they belong
banner("TAG CROPS TO THEIR WINDOW")
crops["stretch"] = -1 # just adding a column where all values -1 - guitly until proven innocent....-1 not in any window
for i, w in my_windows.iterrows(): # hands in  one window row at a time so the first iteration has i = 0 with and w = the window row content, so path, fish_id, frame numeber and stored as series ---so e.g  w.frame_start=300
    crop_inside_window = (crops.frame_number >= w.frame_start) & (crops.frame_number <= w.frame_end) # checking if crop frame is inside the curated window
    crops.loc[crop_inside_window, "stretch"] = i #  df.loc[ WHICH_ROWS , WHICH_COLUMNS # here each crop gets assigned to which window in the video it belongs with the value i from the loop
kept = crops[crops.stretch != -1] # filtering out the crops outside the window
banner_sub(f"kept {len(kept)} of {len(crops)} crops inside verified windows")
logger.info(kept.groupby("stretch").size().to_string())
banner_sub(f"dataframe update")
logger.info(kept.head().to_string())


# selecting the crops from each stretch
banner(f"SUBSAMPLE across all stretches FOR DIVERSITY")
STEP = 15 # keep every 15th crop
curated = (kept.sort_values("frame_number") # Sort the whole table so frames go in order: 298, 299, 3
                .groupby(["stretch", "fish_id"], group_keys=False) # 10 streches x 5 fish (ids 1 to 5) - 50 combnination
                .apply(lambda g: g.iloc[::STEP])) # apply loops of the 50 grouped buckets - glues every function run together.... # [start : stop : step] # step is 15 so take evey 15 crop from each window
banner_sub(f"subsampled to {len(curated)} of {len(kept)} crops")
logger.info(curated.drop(columns='path').head(4).to_string())                                   # first few rows
logger.info(curated.groupby(["stretch", "fish_id"]).size().to_string())   # per-bucket counts

# saving on disk the selected crops
banner("COPY CURATED CROPS OUT")

curated_root = os.path.join(run_dir, "curated_crops") # # output_fish_tracker/<run>/curated_crops
logger.info(f"copying into {curated_root}")
copied = 0
for i, row in curated.iterrows():
    dest_folder = os.path.join(curated_root, f"stretch{row.stretch:02d}_fish{row.fish_id}") #stretch00_fish1/ with 02d d fromat decimal integer, 2...at least two charcaters...0 pad empty space leading zeros...05..
    os.makedirs(dest_folder, exist_ok=True) # ensuring folder exist...if not then create.
    shutil.copy(row.path, dest_folder) # copy using the path all the files in the destination folder
    copied += 1
banner_sub(f"copied {copied} of {len(curated)} crops into {curated_root}")

