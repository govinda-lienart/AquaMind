
# IMPORTS

import pandas as pd
import sys
import logging

# LOGGING

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Helper function

def dry_run():
    ground_truth = pd.DataFrame([ 
        {"frame_start": 790, "frame_end": 860, "fish_ids": "1,3", "error_type": "crossing"}, 
    ])


    tracker_events = pd.DataFrame([ # preditions of fish tracker - see log
            {"frame_number": 800, "event_type": "crossing", "fish_ids": "1,3", "tracker_decision": "no_swap"},  # match ✅  frame in-range, same fish  - This is the ORACLE
            {"frame_number": 900, "event_type": "crossing", "fish_ids": "1,3", "tracker_decision": "no_swap"},  # no match 🚫  right fish, frame outside
            {"frame_number": 850, "event_type": "crossing", "fish_ids": "2,4", "tracker_decision": "no_swap"},  # no match 🚫  frame inside, wrong fish
        ])


    logger.info("TEST MODE")

# Main function

def main():
    logger.info("REAL MODE")

# IF NAME PART

if __name__ == '__main__':
    if "--dry-run" in sys.argv: 
        dry_run()
    else:
        main()

ground_truth = pd.DataFrame([
        {"frame_start": 790, "frame_end": 860, "fish_ids": "1,3", "error_type": "crossing"},
])





