# IMPORTS

import pandas as pd
import logging
import glob # search file names/paths
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)
import re   
import os

# LOAD

# load with all curated frames no swapping"
windows_csv_path = "output_fish_tracker/curation_windows.csv"
windows = pd.read_csv(windows_csv_path,sep=";")
logger.info(windows.head().to_string())
logger.info(windows.shape)

# GATHER CROPS (the exctracted bboxes from the tracker)
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
rows = [] # list of dicsiotnaries
for path in crop_paths:
    m = re.search(r"frame_(\d+)_fish_(\d+)",path)
    frame_num = int(m.group(1)) # the first capture - 298 # frame 298
    fish_id = int(m.group(2)) # second capture - 4 # fish 4
    rows.append({"path":path, "frame_number":frame_num, "fish_id": fish_id}) # dictionary

# converting with pandas rows into in df -? each dict is one row....each key is a column name
crops = pd.DataFrame(rows)
logger.info(crops.head().to_string())
logger.info(crops.shape)