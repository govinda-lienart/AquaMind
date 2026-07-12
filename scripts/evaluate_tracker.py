""" merges and filters ground_data and tracker_events"""


# IMPORTS

import pandas as pd

import argparse
 
# logging object and config
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# TESTING WITH FAKE DATA # when shell command --dry-run

# TEST FUNCTION 

def dry_run():
    """to test the cross join on synthetics data with a know expected result (oracle) before real data"""

    # set up dataframes for fake data ground_truth and tracker_events
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

    # use function to filter out raw of interest using a boolean mast
    
    matched  = crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth) # output matched data row

    # make the script cry/crash if the result not as expected

    matched_frame = set(matched["frame_number"]) # {800} # oracle - from fake event data - found with interval of ground truth data and fish-ids # A set ignores order and index, so i compare just the values cleanly.
    expected_matched_frame = {800}
    assert matched_frame == expected_matched_frame, f"JOIN BRAKE: expected {expected_matched_frame}, but got {matched_frame}" # if false - assert will stop the program
    logger.info(f"---PASSED ASSERTION TEST - matched exactly {expected_matched_frame} --- ")

# HELPER FUNCTION

def crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth):
    """cross pair every tracker event with every ground-truth window"""
    
    # cross merge ground truth with fake data
    pairs = tracker_events.merge(ground_truth, how='cross', suffixes=("_ev", "_gt"))
    logger.debug("--- merged_table ---")
    logger.debug(pairs.to_string())    

    # check boolean fish_ids between ground truth and fake data
    same_fish = pairs["fish_ids_ev"] == pairs["fish_ids_gt"] # creates a frame with a boolean column if true or false fish ids between fake data and ground data is true or false
    logger.debug("--- same_fish_table filter ---")
    logger.debug(same_fish.to_string())

    # check if frame of fake data inside the ground truth data
    frame_inside = pairs["frame_number"].between(pairs["frame_start"], pairs["frame_end"])
    logger.debug("--- frame_number filter:  fake data frame found inside ground_truth_data ---")
    logger.debug(frame_inside.to_string())

    # combining both filters same_fish and frame_inside
    keep = same_fish & frame_inside
    logger.debug("--- applying both filters: same fish AND frame_inside ---")
    logger.debug(keep.to_string())
    
    # use keep to now gram the row values that are true in the dataframe
    matched = pairs[keep] # boolean mask selecting true values
    logger.info ("---  matched data after filter --- ")
    logger.info (matched.to_string())
    return matched

# MAIN FUNCTION # RUNNING ON REAL DATA

def main(events_path, ground_truth_path): # those args are provided by the user in the shel 
    logger.info("REAL MODE")

    # reading csv file produced by parse_tracker_log.py which parsed the output of the fish tracker and converting it into dataframe tracker_events
    tracker_events = pd.read_csv(events_path)
    logger.info(" ---real tracker_events ---")
    logger.info(tracker_events.head().to_string())
    print()

    # reading csv human based csv ground truth and converting into the dataframe ground_truth
    ground_truth = pd.read_csv(ground_truth_path, sep = ";") # my excel generated csv as european format ; which is not the standard comma , thereore need to explitily mention 
    ground_truth["error_type"] = ground_truth["error_type"].str.strip()   # convertion to csv led to some extra spaces after some categories like "occlusion_recovery\t" and therefore creating a different categortu than "occlusion_recovery.
    logger.info (" --- real ground truth - first few rows ---")
    logger.info(ground_truth.head().to_string())


    # cross merging both dataframes tracker_events and ground_truth
    matched = crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth)   # DataFrames
    logger.info("\n --- cross match real data ---\n")
    cols = ["frame_number", "event_type", "fish_ids_ev", "tracker_decision",
            "frame_start", "frame_end", "error_type"]
    logger.info(matched[cols].to_string(index=False)) # here i selected just the columns of interest to make the df smaller and visually easier to look at + index to false removing the first index column                                            

# ENTRY POINT/GUARD + CREATING PARSER

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate tracker events againt human ground truth") # builts the parser object 
    parser.add_argument("events", nargs="?", help="path to the tracker generated log csv")
    parser.add_argument("ground_truth", nargs="?", help="path to the human ground-truth csv")
    parser.add_argument("--dry-run", action="store_true", help="run the synthetic self - test instead of the real data") # store_true --? toggle...if flag (--dry__run) as argument then action store the value args.dry_run = True, if no argument - nothing args.dry_run = False
    args = parser.parse_args() # method from object - read the command line and catch result (arguments typed in by user)
    
    if args.dry_run: # in the shell  --dry-run but converted automaticaly by python as dry_run
        dry_run()
    else:
        main(args.events, args.ground_truth) 







