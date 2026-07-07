
"""Reads fish_tracker.py's raw .log file for one run and extracts every crossing-resolution and occlusion-recovery decision line into a clean,
structured table, to be compared against the human ground-truth table.

Input:  path to a tracker run's .log file 
Output: CSV with columns frame_number | event_type | fish_ids | tracker_decision
"""

import re
import pandas as pd


# patterns for the line types decided to capture from the log and extract
    # pattern 1: one regex for "IDs swapped" / "no swap" lines
    # pattern one regex for "proximity only... skipped" lines
    # one regex for "recovered after N missing frames" lines 
