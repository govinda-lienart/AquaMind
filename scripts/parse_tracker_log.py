"""Reads fish_tracker.py's raw .log file for one run and extracts every crossing-resolution and occlusion-recovery decision line into a clean,
structured table, to be compared against the human ground-truth table.

Input:  path to a tracker run's .log file
Output: CSV with columns frame_number | event_type | fish_ids | tracker_decision
"""

import sys # reading the log path off the command line
import json #  for decoding the structured log lines
import pandas as pd 

EVENT_TYPES = {"crossing", "occlusion_recovery"} # set with unique values and quick reading

def parse_log(log_path):
    with open (log_path, "r") as f:
        for line in f:
            try:
                json.loads(line)
            except json.JSONDecodeError: # belongs to the ValueErrors exception
                continue # silently ignoring all the non json line as most of the lines are json -- otherwise would be very noisy

def main(): 
    log_path = sys.argv[1] # read the argument of the user in the shell command
    df = parse_log(log_path)
    print (len(df))


if __name__ == '__main__':
    main()