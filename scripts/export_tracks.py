import pandas as pd
from scripts.db import get_connection
import argparse
import os

# LOGGER
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

def main(output_dir):

    QUERY = """SELECT frame_number, timestamp, fish_id, x, y, confidence, occluded
    FROM tracks
    ORDER BY frame_number"""

    conn = get_connection()
    df = pd.read_sql(QUERY, conn)
    logger.info(df.head().to_string())
    logger.info(df.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracker events againt human ground truth") #  builts the parser object
    parser.add_argument("--output-dir", 
                        default="output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853",
                        help="folder to save the tracks parquet into")
    args = parser.parse_args # method from object - read the command line and catch result (arguments typed in by user)
    main(args.output_dir)
