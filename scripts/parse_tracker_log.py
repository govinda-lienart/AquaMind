"""Reads fish_tracker.py's raw .log file for one run and extracts every crossing-resolution and occlusion-recovery decision line into a clean,
structured csv file to be compared against the human ground-truth table.

Input:  path to a tracker run's .log file
Output: CSV with columns frame_number | event_type | fish_ids | tracker_decision
"""

# IMPORTS

import sys # reading the log path off the command line
import json #  for decoding the structured log lines
import pandas as pd 
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONFIG

EVENT_TYPES = {"crossing", "occlusion_recovery"} # set with unique values and quick reading
FPS = 60

# HELPER FUNCTIONS

def parse_log(log_path): # {"event": "occlusion_recovery", "fish_ids": "3", "decision": "recovered", "frame": 5432, "missing_frames": 5}
    """this helper function excracts from the log all relevant event (occlusion and crossing) and convert into list of dictionaries"""
    rows = [] 
    with open (log_path, "r") as f:    
        for line in f:
            try:
                obj = json.loads(line) # converting JSON into dictionary
                if obj["event"] in EVENT_TYPES:
                    rows.append({
                        "event_type": obj["event"],
                        "frame_number": obj["frame"],
                        "fish_ids": obj["fish_ids"],
                        "tracker_decision": obj["decision"],
                    })
                    logger.debug(f"captured {obj['event']} at frame {obj['frame']} containing fish{obj['fish_ids']} with decision {obj['decision']}")
            except json.JSONDecodeError: # belongs to the ValueErrors exception
                logger.debug(f"skipped non-JSON: {line.strip()}")
                continue # silently ignoring all the non json line as most of the lines are json -- otherwise would be very noisy
    return rows     

def main(): 
    log_path = sys.argv[1] # grab the log parth in the comnmand shell
    rows = parse_log(log_path)

    # convert list of dictionaries into dataframe
    df = pd.DataFrame(rows)

    # import calibratio value from sidecar'
    sidecar_path = log_path.replace('.log', "_config.yaml")
    with open(sidecar_path) as f:
        cfg = yaml.safe_load(f)
    calibration_secs = cfg['calibration_secs']
    calibration_end_frame = calibration_secs * FPS # fps is 60

    # excluding frames belonging to the calibration secion of the fish tracker
    df = df[df["frame_number"] >= calibration_end_frame]
    logger.debug(f"filtered dataframe ({len(df)} rows):\n{df.head(10)}")

    # convert dataframe to csv
    csv_file_name = log_path.replace(".log","_events.csv")
    df.to_csv(csv_file_name, index=False)

    # summary print
    print(f"SUMMARY: saved {len(df)} events to {csv_file_name}")

if __name__ == '__main__':
    main()
