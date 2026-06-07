
# Setup:

# Import os, datetime, yaml

import os
import datetime
import yaml

# Open config.yaml and load it into a dictionary called cfg

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# Import get_connection from scripts.db

from scripts.db import get_connection

# Open a database connection using get_connection() as a context manager

conn = get_connection()

# Create a cursor for reading called reading_cursor

reading_cursor = conn.cursor()

# Create a cursor for inserting called insert_cursor

inset_cursor = conn.cursor()

# Session:

# Generate session_id by combining the string 'annot_' with the current date and time formatted as YYYYMMDD_HHMM

session_id = "annot_" + datetime.datetime.now().strftime('%Y%M%d_%H%M')

# Print the session ID 

print (session_id)

# Config values:

# Get labels_path from the store_annotations section of config

labels_path = ['store_annotations']['labels_path']

# Get frames_folder from the same section

frames_folder = ['store_annotations']['frames_folder']

# Create a dictionary mapping 0 to "danio_rerio" and 1 to "reflection"

label_map = {0: "danio_rerio", 1: "reflection"}

# List all files in labels_path

listing_labeltxt = os.listdir('labels_path')








# Set three counters to zero: total_frames, total_annotations, total_keypoints
# Loop over each file:

# For each filename in the listing:

# Split the filename on - and take the second part
# Remove the extension, split on _, take the second part and convert to integer — that's the frame number
# Build a search pattern using frames_folder, the frame number, and a % wildcard
# Execute a SELECT query on frames using LIKE to find the matching frame ID
# Fetch the first result and unpack the integer
# Clear the cursor buffer with fetchall
# Increment total_frames
# Print the frame number and frame ID

# Read the annotation file:

# Build the full file path by joining labels_path and the filename
# Open the file and read all lines
# Loop over each line:

# For each line, split on whitespace into a list called values
# If there are 5 values: parse class_id, x_center, y_center, width, height — set has_keypoint = False
# If there are 8 values: parse all 5 bbox values plus kp_x, kp_y, kp_visible — set has_keypoint = True
# Otherwise: print a warning and skip to the next line with continue
# Look up the label name using label_map
# Set created_at to the current datetime

# Insert annotation:

# Execute an INSERT into annotations with all 9 values
# Get the auto-generated ID with lastrowid and store as annotation_id
# Increment total_annotations

# Insert keypoint if applicable:

# If has_keypoint is True AND label is 'danio_rerio':
# Execute an INSERT into keypoints with annotation_id, 'eye', kp_x, kp_y, kp_visible, created_at
# Increment total_keypoints
# Print the keypoint coordinates
# Finish:

# Commit the transaction
# Print the summary block with session ID, created at, labels path, frames folder, and all three counters
# Print the two SQL cross-check queries