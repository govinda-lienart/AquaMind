# IMPORTS

import pandas as pd
import logging
import glob # search file names/paths
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)
import re   

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

# extract the frame number frome one filename (test)