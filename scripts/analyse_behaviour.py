# IMPORTS

import pandas as pd
import argparse
import numpy as np

# LOGGER
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

# MAIN FUNCTION

def main(parquet_path):
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
    parser.add_argument("--parquet-path", 
                        default="output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet",
                        help="path to parquet file")
    args = parser.parse_args() # method from object - read the command line and catch result (arguments typed in by user)
    main(args.parquet_path)



