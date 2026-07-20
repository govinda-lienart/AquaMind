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
    banner_sub('DATAFRAME - SORTED BY FISH_ID AND FRAMENUMBER') 
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

    # binning

    bin_width = 100
    df['x_bin'] = np.floor(df['x'] / bin_width) * bin_width  # (e.g Position 1010: I divide by 100, so it's at position 10.10. I floor it, so now it's 10. Then I multiply, and it belongs to the 1000 bin.)
    depth_profile_all = df.groupby('x_bin')['depth_pct'].mean()
    overall_mean_depth = df['depth_pct'].mean()
    bin_counts = df.groupby('x_bin').size()
    banner_sub("DEPTH PROFILE (mean percentage depth per x_colum, all fish, in cm")    
    logger.info(depth_profile_all.head().to_string())
    logger.info(f"overall mean depth is {overall_mean_depth}")
    banner_sub("bin counts")    
    logger.info(bin_counts.to_string())

    # 2 subplots (percentage depth + frequency for each bin)

     # plot
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]}   # depth panel 3x taller than counts
    )

    # top panel - depth profile
    
    counts, xedges, yedges, mesh = ax_top.hist2d(
        df['x'], df['depth_pct'], bins=[17, 20], cmap='viridis')
    ax_top.plot(depth_profile_all.index, depth_profile_all.values,
                color='white', linewidth=2,
                label='mean depth per bin (all fish pooled)')
    ax_top.axhline(overall_mean_depth, color='red', linestyle='--',
                   label=f'overall mean ({overall_mean_depth:.1f} %)')
    ax_top.invert_yaxis()   # 0 (surface) at top, 100 (gravel) at bottom - matches the real tank
    ax_top.set_ylabel("depth (% of water column,\n0 = surface, 100 = substrate)")
    ax_top.set_title("Tank depth profile and occupancy (all fish, % of water column)")
    ax_top.legend()
    fig.colorbar(mesh, ax=[ax_top, ax_bottom], label='frames per cell')

    # bottom panel - how many frames sit behind each bin to have understanding of frequency across the width of the tank
    ax_bottom.bar(bin_counts.index, bin_counts.values,
                  width=bin_width * 0.9, color='grey')
    ax_bottom.set_ylabel("frames")
    ax_bottom.set_xlabel("horizontal position (x, pixels)")

    path_plot = os.path.join(figure_dir, "depth_profile_all_fish.png")
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**depth profile saved in {figure_dir}**\n")
    plt.close(fig)


    #-------------------
    # Zone Occupancy
    #-------------------
    banner("ZONE OCCUPANCY") 
    
    # splittig the data in 3 depth zones in the tank

    df['zone'] = pd.cut(df['depth_pct'], # cut accepts array
                bins = np.linspace(0,100, 4), # cut in 4 pieces and returns a numpy array  [0, 33.3, 66.7, 100] # bins tells pandas a list of edges/intervalss...cut will cut a range at those points
                labels=['top', 'middle', 'bottom'], # the order is important here ....first inner band 0->33.3 is top, 33.3 -> 66.7 is middle, and 66.7 to 100 is bottom
                include_lowest=True) # include 0 to be able to deal with Nan of first value

    banner_sub("return of np.linspace(0,100, 4)")
    logger.info(np.linspace(0,100,4))
    banner_sub("df zone")
    logger.info(df.head().to_string())             

    # frequncy counts per zone across all fish

    frequency_table = df['zone'].value_counts().sort_index() # sort index presents the data with top, middle , bottom instrad of by values bottom, middle, top
    frequency_pct_table = df['zone'].value_counts(normalize=True).sort_index() # that will give a percentage...like for 15622 bottom/28800 total = 54 percent
    banner_sub("frequency_values")
    logger.info(frequency_table.to_string())   
    banner_sub("frequency_values_percentage")
    logger.info(frequency_pct_table.to_string())          

    # frequncy counts per zone per fish_id
    zone_by_fish = df.groupby('fish_id')['zone'].value_counts(normalize=True)
    banner_sub("ZoONE FRACTIONS - PER FISH (raw)")
    logger.info(zone_by_fish.to_string())

    # stacking to ease the plotting
    unstack_frequency_table = zone_by_fish.unstack() # i need to unstack the rows and create a table # take a level of the index and pivot it into columns - series to frame # data gets wider, index shorter good for plotting but stack() data gets taller, index longer, food for storing and grouping
    
    banner_sub("unstack frequency table")
    logger.info(unstack_frequency_table.to_string())   

    unstack_frequency_table.loc['all'] = frequency_pct_table # append pooled row --> 5 rows #  When you assign a Series into a row with .loc, pandas matches the Series' index against the table's column names and drops each value into the right slot.
    banner_sub("unstack frequency table for each fish_id and accross all fish (loc)")
    logger.info(unstack_frequency_table.to_string())   

    # plotting 

    unstack_frequency_table[['bottom', 'middle', 'top']].plot(
        kind='bar', stacked=True, figsize=(8, 5),
        color=['#1f4e79', '#4d94c4', '#7fc7ff'])
    plt.ylabel("fraction of time('frame)")
    plt.xlabel("fish id")
    plt.title('Vertical zone occupancy per fish')
    plt.xticks(rotation=0)
    plt.legend(title='zone', loc='upper left', bbox_to_anchor=(1.02, 1)) # 1.02 is just past the right-hand edge, and 1 is level with the top / loc = park the legend's top-left corner just outside the plot's right edge
    path_plot = os.path.join(figure_dir, "zone_occupancy_per_fish.png")
    plt.savefig(path_plot, dpi=150, bbox_inches='tight') # tight crop the whitespace around the figure to fit the actual content. 
    logger.info(f"\n**zone occupancy saved in {figure_dir}**\n")
    plt.close()

    #-------------------
    # DEPTH OVER TIME
    #-------------------
    banner("DEPTH OVER TIME")

    seconds = df['timestamp'].astype(int)  # round down to whole seconds

    # one column per fish, one row per second
    depth_per_sec = (df.groupby(['fish_id', seconds])['depth_pct']
                       .mean()
                       .unstack(level=0))

    # group summary that survives averaging: share of fish in the bottom third
    # each second (0 = none, 1 = all). a mean of depths would invent a
    # mid-water position no fish actually occupies
    frac_bottom = df.groupby(seconds)['depth_pct'].apply(lambda s: (s > 66.7).mean())

    # how much of each second is Kalman prediction rather than real detection
    occluded_share = df.groupby(seconds)['occluded'].mean()

    banner_sub("DEPTH PER SECOND PER FISH")
    logger.info(depth_per_sec.head().to_string())

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]})

    # top - each fish's own depth trace
    depth_per_sec.plot(ax=ax_top, linewidth=1.2)
    ax_top.axhline(66.7, color='grey', linestyle='--', label='bottom-third boundary')
    ax_top.axhline(33.3, color='grey', linestyle=':', label='top-third boundary')
    ax_top.set_ylim(100, 0)
    ax_top.set_ylabel("depth (% of water column,\n0 = surface, 100 = substrate)")
    ax_top.set_title("Depth over time per fish")
    ax_top.legend(title='fish id', loc='upper left', bbox_to_anchor=(1.02, 1))

    # bottom - group summary + how much of it rests on predicted positions
    ax_bottom.fill_between(frac_bottom.index, frac_bottom.values,
                           color='#1f4e79', alpha=0.6, label='share in bottom third')
    ax_bottom.plot(occluded_share.index, occluded_share.values,
                   color='orange', linewidth=1, label='share occluded')
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_ylabel("share of fish")
    ax_bottom.set_xlabel("time (seconds)")
    ax_bottom.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    path_plot = os.path.join(figure_dir, "depth_over_time_per_fish.png")
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**depth over time saved in {figure_dir}**\n")
    plt.close(fig)

    #-------------------
    # ACTIVITY - GRID LINE CROSSINGS
    #-------------------
    banner("ACTIVITY - LINE CROSSINGS")

    grid_cm = 4                              # ~1 adult zebrafish body length - the standard heuristic
    grid_px = grid_cm * pixels_per_cm        # 4 cm -> ~150 px at this calibration
    logger.info(f"grid: {grid_cm} cm = {grid_px:.0f} px per cell")

    # which grid cell is each fish in, this frame (same floor-divide trick as x_bin)
    df['cell_x'] = np.floor(df['x'] / grid_px)
    df['cell_y'] = np.floor(df['y'] / grid_px)

    # how many grid LINES were crossed between this frame and the last, per fish.
    # abs() of the cell change counts each boundary: a diagonal move that changes
    # both cell_x and cell_y crosses 2 lines, and a fast move spanning 3 cells counts 3
    grouped_fish = df.groupby('fish_id')
    df['lines_crossed'] = (grouped_fish['cell_x'].diff().abs().fillna(0)
                         + grouped_fish['cell_y'].diff().abs().fillna(0))

    # total per fish = the activity score
    crossings_per_fish = df.groupby('fish_id')['lines_crossed'].sum()

    # rate, so sessions of different length stay comparable
    session_secs = df['timestamp'].max() - df['timestamp'].min()
    crossings_per_min = crossings_per_fish / session_secs * 60

    banner_sub("LINE CROSSINGS PER FISH")
    logger.info(crossings_per_fish.to_string())
    banner_sub("LINE CROSSINGS PER MINUTE")
    logger.info(crossings_per_min.to_string())

    # per second, for the time course
    crossings_per_sec = (df.groupby(['fish_id', 'second'])['lines_crossed']
                           .sum()
                           .unstack(level=0))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})

    # total activity score per fish
    crossings_per_min.plot(kind='bar', ax=ax_top, color='#4d94c4')
    ax_top.set_ylabel("lines crossed\nper minute")
    ax_top.set_xlabel("fish id")
    ax_top.set_title(f"Activity - grid line crossings ({grid_cm} cm grid)")
    ax_top.tick_params(axis='x', rotation=0)

    # time course - this is where a pre/post stimulus drop would show up
    crossings_per_sec.plot(ax=ax_bottom, linewidth=1.2)
    ax_bottom.set_ylabel("lines crossed per second")
    ax_bottom.set_xlabel("time (seconds)")
    ax_bottom.legend(title='fish id', loc='upper left', bbox_to_anchor=(1.02, 1))

    path_plot = os.path.join(figure_dir, "activity_line_crossings.png")
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    logger.info(f"\n**line crossings saved in {figure_dir}**\n")
    plt.close(fig)



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

 

