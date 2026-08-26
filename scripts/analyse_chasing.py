"""usage e.g python -m scripts.analyse_chasing --video_name IMG_2349"""

#---------------
# IMPORTS
#---------------

import os
import pandas as pd
import argparse

import matplotlib
matplotlib.use('Agg') # avoids popup windows of poduced plots
import matplotlib.pyplot as plt
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name, trim_to_calibration
from scripts.chasing_features import build_pairs
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

#---------------
# MAIN FUNCTION
#---------------


def main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end):

    banner('COMPUTING')
    df = pd.read_parquet(parquet_path)
    banner_sub('LOADING PARQUET FILE')
    logger.info(f'{df.head().to_string()}')

    df = trim_to_calibration(df, calibration_secs, frame_number_end)
    pairs = build_pairs(df, pixels_per_cm)

    banner('OUTPUT GRAPHS')
    output_folder = os.path.dirname(parquet_path) # parquet_path = 'output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853/tracks.parquet' # dirname() strips the filename off, leaving just the folder:# output_folder = 'output_fish_tracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1853'
    figure_dir = os.path.join(output_folder, "output_analyse_chasing")
    os.makedirs(figure_dir, exist_ok=True)

    banner('DISTANCE + CLOSING SPEED OVER TIME PER PAIR')
    for (fish_a, fish_b), pair_rows in pairs.groupby(['fish_id_a', 'fish_id_b']):
        fig, (ax_top, ax_raw, ax_mid, ax_bottom) = plt.subplots(
            4, 1, sharex=True, figsize=(14, 11),
            gridspec_kw={'height_ratios': [2, 1, 1, 1]})

        ax_top.plot(pair_rows['timestamp_a'], pair_rows['distance_cm_smooth'], linewidth=0.8, color='#1f4e79')
        ax_top.set_ylabel("distance (cm)")
        ax_top.set_title(f"Pairwise distance + closing speed, raw vs window=5 vs window=15 — fish {fish_a}-{fish_b}")

        # raw (no smoothing) — its own autoscaled range, since its spikes (+/-500-700) dwarf
        # the smoothed versions; this panel is what shows the "starting point" of the noise problem
        ax_raw.plot(pair_rows['timestamp_a'], pair_rows['closing_speed_cm_s_raw'], linewidth=0.8, color='#a83232')
        ax_raw.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_raw.set_ylabel("closing speed\nraw (cm/s)")

        # window=5 kept on a fixed +/-100 scale, matching its own known noise range
        ax_mid.plot(pair_rows['timestamp_a'], pair_rows['closing_speed_cm_s'], linewidth=0.8, color='#c46210')
        ax_mid.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_mid.set_ylabel("closing speed\nwindow=5 (cm/s)")
        ax_mid.set_ylim(-100, 100)

        # window=15 left on its own autoscaled range (NOT matched to window=5 anymore) so its
        # smaller spikes stretch to fill the panel and are easier to read individually
        ax_bottom.plot(pair_rows['timestamp_a'], pair_rows['closing_speed_cm_s_w15'], linewidth=0.8, color='#2e8b57')
        ax_bottom.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_bottom.set_ylabel("closing speed\nwindow=15 (cm/s)")
        ax_bottom.set_xlabel("time (s)")

        path_plot = os.path.join(figure_dir, f"pairwise_distance_closingspeed_window_compare_{fish_a}_{fish_b}.png")
        fig.savefig(path_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
    logger.info(f"\n**pairwise distance + closing speed plots saved in {figure_dir}**\n")


#---------------
# ENTRY POINT
#---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imports video configs")
    parser.add_argument("--video_name",
                        default="IMG_2349",
                        help="indicate which video you want to analyse")
    args = parser.parse_args()
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end)

 
# APPENDIX 

# calculation distance

"""                 fish A (x_a, y_a)
                         *
                         |\
                         | \
              dy = y_a-y_b \   <- straight-line distance
                         |   \     = hypotenuse
                         |    \
                         |     \
                         *------* 
                                 fish B    (x_a, y_b)  <- imaginary corner point
                    (x_b, y_b)
                       dx = x_a - x_b"""


# calculation closing speed
# CLOSING SPEED — worked examples (real ~60fps data, delta_t ~= 0.01667s per frame)
#
# when fish get closer, distance goes from 10 (previous frame) to 8 (this frame) = -2,
# so delta_distance_cm is negative. flipping the sign makes closing_speed positive
# when fish are approaching — more intuitive to read.
#
# 1) SLOW APPROACH
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 199    3.31667    5.00         —                  —         —
# 200    3.33333    4.95         -0.05               0.01667   -(-0.05)/0.01667 = 3.0
#
# a fish closing the gap by just 0.05cm in one frame (sub-millimeter, barely visible)
# scales up to a closing_speed of 3.0 cm/s once expressed "per second" —
# because 0.05cm x 60 frames/sec ~= 3 cm/s
#
# 2) FAST APPROACH / BURST
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 250    4.16667    3.20         —                  —         —
# 251    4.18333    3.05         -0.15               0.01667   -(-0.15)/0.01667 = 9.0
#
# a bigger per-frame drop (0.15cm instead of 0.05cm) scales up to a closing_speed
# of 9.0 cm/s — three times faster than the slow-approach example above
#
# 3) SEPARATING
# frame  timestamp  distance_cm  delta_distance_cm  delta_t   closing_speed_cm_s
# 300    5.00000    4.00         —                  —         —
# 301    5.01667    4.10         +0.10               0.01667   -(+0.10)/0.01667 = -6.0
#
# distance grew by 0.10cm this frame, so delta_distance_cm is POSITIVE — flipping
# the sign turns that into a NEGATIVE closing_speed, meaning "moving apart, not
# approaching." this is why the sign convention matters: positive closing_speed
# = closing in (candidate chase), negative closing_speed = separating.