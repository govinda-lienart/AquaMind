""" merges and filters ground_data and tracker_events"""


# IMPORTS

import pandas as pd
import sys
import logging

# LOGGING

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# TEST FUNCTION

def dry_run():
    """to test the cross join on synthetics data with a know expected result (oracle) before real data"""

    ground_truth = pd.DataFrame([ 
        {"frame_start": 790, "frame_end": 860, "fish_ids": "1,3", "error_type": "crossing"}, 
    ])

    tracker_events = pd.DataFrame([ # preditions of fish tracker - see log
            {"frame_number": 800, "event_type": "crossing", "fish_ids": "1,3", "tracker_decision": "no_swap"},  # match ->  frame in-range, same fish  - This is the ORACLE
            {"frame_number": 900, "event_type": "crossing", "fish_ids": "1,3", "tracker_decision": "no_swap"},  # no match ->  right fish, frame outside
            {"frame_number": 850, "event_type": "crossing", "fish_ids": "2,4", "tracker_decision": "no_swap"},  # no match ->  frame inside, wrong fish
     ])

    logger.info("--- ground_truth ___")
    logger.info(ground_truth.to_string()) # to_string - dataframe method to turn it into a nice table visually
    logger.info("--- tacker events ---") 
    logger.info(tracker_events.to_string())

    crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth)

# HELPER FUNCTION

def crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth):
    """cross pair every tracker event with every ground-truth window"""
    
    # cross merge ground truth with fake data
    pairs = tracker_events.merge(ground_truth, how='cross', suffixes=("_ev", "_gt"))
    logger.info("--- merged_table ---")
    logger.info(pairs.to_string())    

    # check boolean fish_ids between ground truth and fake data
    same_fish = pairs["fish_ids_ev"] == pairs["fish_ids_gt"] # creates a frame with a boolean column if true or false fish ids between fake data and ground data is true or false
    logger.info("--- same_fish_table filter ---")
    logger.info(same_fish.to_string())

    # check if frame of fake data inside the ground truth data
    frame_inside = pairs["frame_number"].between(pairs["frame_start"], pairs["frame_end"])
    logger.info("--- frame_number filter:  fake data frame found inside ground_truth_data ---")
    logger.info(frame_inside.to_string())

    # combining both filters same_fish and frame_inside
    keep = same_fish & frame_inside
    logger.info("--- applying both filters: same fish AND frame_inside ---")
    logger.info(keep.to_string())


    # use keep to now gram the row values that are true in the dataframe
    
    return keep

# MAIN FUNCTION

def main():
    logger.info("REAL MODE")

# ENTRY POINT/GUARD

if __name__ == '__main__':
    if "--dry-run" in sys.argv: 
        dry_run()
    else:
        main()






