""" merges and filters ground_data and tracker_events"""


# IMPORTS

import pandas as pd

import argparse
 
# logging object and config
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def banner(title):
    """print a loud section header to the console so the flow is easy to follow"""
    logger.info("\n" + "═" * 78)
    logger.info(f"  {title}")
    logger.info("═" * 78)

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
    logger.debug ("---  matched data after filter --- ")
    logger.debug(matched.to_string())
    return matched

# MAIN FUNCTION # RUNNING ON REAL DATA

def main(events_path, ground_truth_path): # those args are provided by the user in the shell command
    logger.info("REAL MODE")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LOAD INPUTS  (tracker's log + human's review)
    # ════════════════════════════════════════════════════════════════════════
    banner("SECTION 1 — LOAD INPUTS")

    # 1a. tracker events = csv produced by parse_tracker_log.py (one row per risky decision the tracker logged)
    tracker_events = pd.read_csv(events_path) 
    event_cols = ["frame_number", "event_type", "fish_ids", "tracker_decision"] # selecting key relevant coluns for better visibility
    logger.info(f"\n[1a] tracker_events loaded: {len(tracker_events)} rows  (each row = one risky decision the tracker logged)\n")
    logger.info(tracker_events[event_cols].to_string())

    # 1b. ground truth = human review csv (one row per observed error window)
    ground_truth = pd.read_csv(ground_truth_path, sep = ";") # excel exports european-format csv (; not ,), so declare the separator
    ground_truth["error_type"] = ground_truth["error_type"].str.strip()   # csv export left trailing spaces/tabs (e.g. "occlusion_recovery\t") that would split one category into two
    ground_truth["gt_id"] = range(len(ground_truth))   # stable id per ground-truth row , that way we can find which exceptions the tracker never matched

    gt_cols = ["frame_start", "frame_end", "error_type", "gt_id"] # selecting relevant columns for better visbility
    logger.info(f"\n[1b] ground_truth loaded: {len(ground_truth)} rows  (each row = one error window I observed; gt_id is unique here)\n")
    logger.info(ground_truth[gt_cols].to_string(index=False)) # the false removes the auto increment column

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — CROSS-MATCH  (pair each tracker event with any ground truthed window it falls inside)
    # ════════════════════════════════════════════════════════════════════════
    banner("SECTION 2 — CROSS-MATCH events to human windows")

    matched = crossmatch_tracker_events_to_ground_truth(tracker_events, ground_truth)   # every event × any containing window ground truth, same fish
    cols = ["frame_number", "event_type", "fish_ids_ev", "tracker_decision",
            "frame_start", "frame_end", "error_type", "gt_id"]
    logger.info(f"\n[2] matched: {len(matched)} rows kept out of {len(tracker_events)}×{len(ground_truth)} = {len(tracker_events)*len(ground_truth)} possible pairs")
    logger.info(" crossmatch function allows a row to survive only if same fish AND event frame inside the window -> gt_id repeats where one window caught several events)\n")
    logger.info(matched[cols].to_string(index=False)) # only the columns of interest + index=False to drop the row-index column

    # switches = matched pairs where the tracker's event_type agrees with the human's error_type (a confirmed ID switch)
    switches = matched[matched["event_type"] == matched["error_type"]]
    logger.info(f"\n[2] switches: {len(switches)} of {len(matched)} matched rows survive the cause-agreement filter (tracker cause == human cause)\n")
    logger.info(switches[cols].to_string(index=False)) # swiches refers to when identity error really happened here"

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — HUMAN-CENTRIC VIEW  ("of my 17 observations, which did the tracker corroborate?")
    #   counts human rows — each gets exactly one disposition, so nothing vanishes
    # ════════════════════════════════════════════════════════════════════════
    banner("SECTION 3 — HUMAN-CENTRIC VIEW: one disposition per observed row")

    # in `matched` a gt_id can repeat — one window (e.g. 1178–1264) caught 4 events, so gt_id 1 appears 4×;
    # a set collapses those repeats back to "which human rows were touched at all"
    matched_ids = set(matched["gt_id"])   # human rows the tracker logged *something* for
    aligned_ids = set(switches["gt_id"])  # subset: human rows that got at least one *cause-matching* event  (aligned_ids and matched_ids)
    logger.info(f"\n[3] collapsing gt_id from the tables above into unique sets of human rows:")
    logger.info(f"     matched_ids ({len(matched_ids)} rows touched)     : {sorted(matched_ids)}")
    logger.info(f"     aligned_ids ({len(aligned_ids)} rows corroborated): {sorted(aligned_ids)}   (always a subset of matched_ids)")

    def disposition(gt_id):
        if gt_id in aligned_ids:      # touched AND cause matches
            return "confirmed_switch"
        if gt_id in matched_ids:      # touched, but cause doesn't match
            return "matched_mismatch"
        return "unmatched"            # never touched at all

    ground_truth["disposition"] = ground_truth["gt_id"].apply(disposition)   # stamp one label on every human row

    logger.info(f"\n[3] every one of the {len(ground_truth)} human rows now carries exactly one disposition (nothing dropped):\n")
    logger.info(ground_truth[gt_cols + ["disposition"]].to_string(index=False))

    # accounting: how many human rows fell into each bucket
    disposition_counts = ground_truth.groupby("disposition").size()
    logger.info("\n[3] human rows by disposition:\n")
    logger.info(disposition_counts.to_string())

    # self-check: every human row lands in exactly one bucket, so the counts must sum to the total (else a row was lost)
    assert disposition_counts.sum() == len(ground_truth), \
        f"ACCOUNTING BROKE: buckets sum to {disposition_counts.sum()}, but there are {len(ground_truth)} human rows"
    logger.info(f"\n[3] ✓ accounting OK: {disposition_counts.sum()} rows across {len(disposition_counts)} buckets = {len(ground_truth)} total (invariant holds)")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — TRACKER-CENTRIC VIEW  ("of the tracker's decisions, what fraction were confirmed wrong?")
    #   counts events — one wide window can contribute several (per-decision rate, not per-observation)
    # ════════════════════════════════════════════════════════════════════════
    banner("SECTION 4 — TRACKER-CENTRIC VIEW: ID-switch rate over events")

    logger.info(f"\n[4] re-using the {len(switches)} confirmed switches printed in Section 2, now counted as rates:")

    logger.info("\n[4] confirmed switches by cause (counting events):\n")
    logger.info(switches.groupby("error_type").size().to_string())

    logger.info("\n[4] confirmed switches by fish pair (which fish trip the tracker most):\n")
    logger.info(switches.groupby("fish_ids_ev").size().to_string())

    rate = len(switches) / len(tracker_events) # e.g. 11 confirmed switches / 279 logged events
    logger.info(f"\n[4] overall ID-switch rate: {len(switches)}/{len(tracker_events)} = {rate:.1%}  (fraction of the tracker's risky decisions that were confirmed ID errors)")

    logger.info("\n[4] ID-switch rate by cause (per-cause numerator / denominator, as %):\n")
    switches_by_cause = switches.groupby("event_type").size()        # numerators   e.g. crossing=10, occlusion_recovery=1
    events_by_cause   = tracker_events.groupby("event_type").size()  # denominators e.g. crossing=59, occlusion_recovery=220
    rate_by_cause     = switches_by_cause / events_by_cause          # per-cause rate, e.g. 10/59 ≈ 17% of crossings were confirmed errors
    logger.info((rate_by_cause * 100).round(1).to_string())

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







