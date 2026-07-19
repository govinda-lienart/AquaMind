# IMPORTS

from turtle import distance
from click import group
from matplotlib import legend
import pandas as pd
import argparse
import numpy as np
import yaml
import os

# MATPLOTLIB

import matplotlib
matplotlib.use('Agg') # means no pop up window
import matplotlib.pyplot as plt

# LOGGER

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

# BANNER

def banner(title):
    """print a loud section header to the console so the flow is easy to follow"""
    logger.info("\n" + "═" * 78)
    logger.info(f"  {title}")
    logger.info("═" * 78 + "\n") 

def banner_sub(title):
    """prints description of subdivistion"""
    logger.info(f"\n--- {title} ---\n")

# CONFIG

#calibration to cm using tank size in cm/pixels - to allow relative comparision between studies and fish tanks
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']

# HELPER FUNCTION

def grab_video_name(video_name):
    "grabs arguments from user and pulls out the related parameters from config.yaml"
    video_cfg = cfg['videos'][video_name]
    parquet_path = video_cfg['parquet_path']
    tank_width_px = video_cfg['tank_width_px']
    tank_width_cm = video_cfg['tank_width_cm']
    calibration_secs = video_cfg['calibration_secs']
    surface_y_px = video_cfg['surface_y_px']
    bottom_y_px = video_cfg['bottom_y_px']
    pixels_per_cm = tank_width_px / tank_width_cm
    banner('LOADING CONFIGURATION')
    logger.info(f'loaded cfg: video_cfg = {video_cfg}, tank_width_px = {tank_width_px}, tank_width_cm = {tank_width_cm},  pixels_per_cm = {pixels_per_cm}, calibration_secs = {calibration_secs}, surface_y_px = {surface_y_px}, bottom_y_px = {bottom_y_px}')
    return parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px

# MAIN FUNCTION

def main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px):
    """calculate ruled based behaviour"""

    #---------
    # PARQUET
    #---------

    banner('VISUALING PARQUET CONTENT')

     
    # pulls parquet data into dataframe
    df = pd.read_parquet(parquet_path)
    
    # remove the calibration time - 10 seconds from the dataframe
    df = df[df['timestamp'] >= calibration_secs] # calibration trim - sed here boolean mask to select only those that are true df[true/false]

    # calbirate depth to cm (sqauare pixels)
    df['y_cm'] = df['y'] / pixels_per_cm  # to calculate depth profiles in cm
    df['depth_pct'] = (df['y'] - surface_y_px) / (bottom_y_px - surface_y_px) * 100 # percentage depth of the fish ->  how far the fish is below the surface, divided by how far the bottom is below the surface, times 100."
    # logger parquet

    banner_sub('DATAFRAME - HEAD')
    logger.info(df.head().to_string())
    banner_sub('DATAFRAME - SHAPE')
    logger.info(df.shape)
    banner_sub('DATAFRAME - DESCRIBE')
    logger.info(df.describe().to_string())
    banner_sub('DATAFRAME - INFO')
    logger.info(df.info())

    # output folder for all behaviour figures (build once)
    output_folder = os.path.dirname(parquet_path) # will find output folder based on the prvoided parquet file
    figure_dir = os.path.join(output_folder, "output_analyse_behaviour")    # will create in the output folder a subfolder called output_analysys_behavour       
    os.makedirs(figure_dir, exist_ok=True)   # standalone funciton 

    #-------------
    # FISH SPEED
    #-------------

    banner('FISH SPEED')

    # sort df on fish id and frames
    df = df.sort_values(['fish_id', 'frame_number'])
    banner('DATAFRAME - SORTED BY FISH_ID AND FRAMENUMBER') 
    logger.info(df.head().to_string())

    # sort fish in groups by fish id - and calclate distance swom across x and y as
    grouped_fish = df.groupby('fish_id') # grouped object containing different bags of fish by fish_id
    df['dx'] = grouped_fish['x'].diff() # calculate difference in x value between present row and preivious one
    df['dy'] = grouped_fish['y'].diff()
    df['dt']= grouped_fish['timestamp'].diff()
    banner_sub("DATAFRAME - DIFFERENCE IN X, Y and TIME BY GROUP FISH_ID")
    logger.info(df.head().to_string())

    # calculating distance - diagonal - pythogoras  √(dx² + dy²).
    df['distance'] = np.hypot(df['dx'], df['dy']) # calculating pythagoras using numpy
    df['distance'] = df['distance'] / pixels_per_cm # here i convert with calibrated number obtained from config  -  divide because each pixel is 1/51.7 of a cm, so you're scaling down. example if a fix move 9.01 pixes then for pixels_per_cm = 1811/35 (pixel tank widht/cm tank width) ≈ 51.7, so 9.01/51.7 ≈ 0.174 cm

    # calculating speed 
    df['speed'] = df['distance'] / df['dt']
    banner_sub("DATAFRAME- WITH NEW COLUMN DISTANCE AND SPEED")
    logger.info(df.head().to_string())

    # calculating - Individual fish mean swimming speed (per second) - mean fish speed cm/sec 
    df['second'] = np.floor(df['timestamp'])
    group_sec_fish = df.groupby(['fish_id','second']) # different bag for different fish...and each bag subbags per secondd
    mean_fish_speed_sec = group_sec_fish['speed'].mean().reset_index() # series -> dataframe thx to reset_index - helps to create dataframe and also flattens index 
    banner_sub("DATAFRAME - MEAN_FISH_SPEED_SEC")
    logger.info(mean_fish_speed_sec.head().to_string())
    logger.info(type(mean_fish_speed_sec))

    #creating the plot  -  Individual fish mean swimming speed (per second)
    plt.figure(figsize=(14, 6))
    for fish_id, fish_rows in mean_fish_speed_sec.groupby('fish_id'): # fish_id is the key: a clean scalar(single value)  → 1  # #
        plt.plot(fish_rows['second'], fish_rows['speed'], linestyle='--', label=f'fish {fish_id}')     # The loop runs once per fish (per fish_id), and each time it plots that fish's entire line in one go —> not row by row - 5 iterations in total
    plt.xlabel("time(s)")
    plt.ylabel("mean speed (cm/s)")
    plt.title("Individual fish mean swimming speed per sec")
    plt.legend()
    path_plot = os.path.join(figure_dir, "fish_speed_mean_per_sec.png")
    plt.savefig(path_plot)

    logger.info(f"\n**speed plot saved in {figure_dir}\n**")
    plt.close()

    # calculating  -  Mean Individual fish mean swimming speed over the entire timeframe
    group_fish = df.groupby(['fish_id'])
    mean_fish_speed = group_fish['speed'].mean() # we can, but no real need here for reset_index() df,  time-series is fine here because bar chart
    banner_sub("TIME-SERIES - MEAN_FISH_SPEED_SEC")
    logger.info(mean_fish_speed.to_string())
    logger.info(type(mean_fish_speed))

    # histogram - Mean Individual fish mean swimming speed over the entire timeframe
    plt.figure(figsize=(14, 6))
    plt.bar(mean_fish_speed.index, mean_fish_speed.values)
    plt.xticks(mean_fish_speed.index)
    plt.xlabel("fish_id")
    plt.ylabel("mean speed (cm/s)")
    plt.title("Individual fish mean swimming over the entire timeframe ")
    path_plot = os.path.join(figure_dir, "fish_speed_mean.png")
    plt.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**histogram saved in {figure_dir}**\n")
    plt.close()

   #------------------------
   # CUMULATIVE DISTANCE SWUM
   #------------------------

    banner('CUMMULATIVE DISTANCE')

    # calculation of cummulative distance

    df['cum_distance']= df.groupby('fish_id')['distance'].cumsum() # distance was already calculated when caclulating speed

    banner_sub("DATAFRAME - COLUMN WITH CUM_DISTANCE")
    logger.info(df.head().to_string())
    logger.info(df.tail().to_string())
    banner_sub("FISH_ID, TIMESTAM, CUM_DISTANCE")
    logger.info(df[['fish_id', 'timestamp', 'cum_distance']].tail().to_string())
    total_distance_per_fish = df.groupby('fish_id')['cum_distance'].max()
    banner_sub("TOTAL DISTANCE SWUM PER FISH (CM)")
    logger.info(total_distance_per_fish.to_string())

    # plot cummulative distance

    plt.figure(figsize=(14,6))
    plt.bar(total_distance_per_fish.index, total_distance_per_fish.values)
    plt.xticks(total_distance_per_fish.index)
    plt.xlabel("fish_id")
    plt.ylabel("total distance swum(cm)")
    plt.title("Total distance swum by individual fish during the 120 seconds video lenght")
    path_plot = os.path.join(figure_dir, "fish_total_distance.png")
    plt.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**histogram saved in {figure_dir}**\n")
    plt.close()

    #------------------------
    # BOTTOM DWELLING AND DEPTH PROFILE
    #----------------------
    banner('DEPTH PROFILE ACROSS THE TANK')

    # bin x pixels in full numbers 

    bin_width = 100
    df['x_bin'] = np.floor(df['x'])
    depth_profile_all = df.groupby('x_bin')['depth_pct'].mean()
    overall_mean_depth = df['depth_pct'].mean()

    banner_sub("DEPTH PROFILE (mean percentage depth per x_colum, all fish, in cm")    
    logger.info(depth_profile_all.head().to_string())
    logger.info(f"overall mean depth is {overall_mean_depth}")

    # plot

    plt.figure(figsize=(14, 6))
    plt.plot(depth_profile_all.index, depth_profile_all.values, label='mean depth across all fish')
    plt.axhline(overall_mean_depth, color='red', linestyle='--',
                label=f'overall mean ({overall_mean_depth:.1f} %)')
    plt.gca().invert_yaxis()
    plt.xlabel("horizontal position (x, pixels)")
    plt.ylabel("mean depth (% of water colunm, 0 = surface, 100 = substrate)")
    plt.title("Tank depth profile (all fish, % of water column)")
    plt.legend()
    path_plot = os.path.join(figure_dir, "depth_profile_all_fish.png")



    plt.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**% depth profile saved in {figure_dir}**\n")
    plt.close()

#-------------------
# ENTRY POINT/GUARD
#-------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imports Parquet snapshot") #  builts the parser object
    parser.add_argument("--video_name", 
                        default="IMG_2349",
                        help="path to video configuration in config.yaml")
    args = parser.parse_args() # method from object - read the command line and catch result (arguments typed in by user)
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px = grab_video_name(args.video_name)
    main(parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px)

 

