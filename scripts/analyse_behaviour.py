import pandas as pd
import argparse

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
    
# ENTRY POINT/GUARD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imports Parquet snapshot") #  builts the parser object
    parser.add_argument("--parquet-path", 
                        default="output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet",
                        help="path to parquet file")
    args = parser.parse_args() # method from object - read the command line and catch result (arguments typed in by user)
    main(args.parquet_path)



