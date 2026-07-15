# IMPORTS

import pandas as pd
import argparse
import numpy as np
import yaml

# LOGGER
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

# CONFIG
#calibration to cm using tank size in cm/pixels - to allow relative comparision between studies and fish tanks
logger.info("\n--- loading configuration---\n")
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']

def grab_video_name(video_name):
    "grabs arguments from user and pulls out the related parameters from config.yaml"
    video_cfg = cfg['videos'][video_name]
    parquet_path = video_cfg['parquet_path']
    tank_width_px = video_cfg['tank_width_px']
    tank_width_cm = video_cfg['tank_width_cm']
    pixels_per_cm = tank_width_px / tank_width_cm
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = { pixels_per_cm} ')
    return parquet_path, pixels_per_cm
  

# MAIN FUNCTION

def main(parquet_path, pixels_per_cm):
    """calculate ruled based behaviour"""

    # pulls parquet data into dataframe
    df = pd.read_parquet(parquet_path)
    logger.info("\n--- dataframe - read head ---\n") 
    logger.info(df.head().to_string())
    logger.info("\n--- dataframe - shape ---\n") 
    logger.info(df.shape)
    logger.info("\n--- dataframe - describe ---\n") 
    logger.info(df.describe().to_string())
    logger.info("\n--- dataframe - info ---\n") 
    logger.info(df.info())

    #-------------
    # FISH SPEED
    #-------------

    # sort df on fish id and frames
    df = df.sort_values(['fish_id', 'frame_number'])
    logger.info("\n--- dataframe - sorted by fish_id and framenumber ---\n") 
    logger.info(df.head().to_string())

    # sort fish in groups by fish id - and calclate distance swom across x and y as
    grouped_fish = df.groupby('fish_id') # grouped object containing different bags of fish by fish_id
    df['dx'] = grouped_fish['x'].diff() # calculate difference in x value between present row and preivious one
    df['dy'] = grouped_fish['y'].diff()
    df['dt']= grouped_fish['timestamp'].diff()
    logger.info("\n--- dataframe - difference in x, y and time by group fish_id\n")
    logger.info(df.head().to_string())

    # calculating distance - diagonal - pythogoras  √(dx² + dy²).
    df['distance'] = np.hypot(df['dx'], df['dy']) # calculating pythagoras using numpy

    # calculating speed 
    df['speed'] = df['distance'] / df['dt']
    logger.info("\n--- dataframe - with new columns distance and speed\n")
    logger.info(df.head().to_string())


# ENTRY POINT/GUARD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imports Parquet snapshot") #  builts the parser object
    parser.add_argument("--video_name", 
                        default="IMG_2349",
                        help="path to video configuration in config.yaml")
    args = parser.parse_args() # method from object - read the command line and catch result (arguments typed in by user)
    parquet_path, pixels_per_cm = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm)



