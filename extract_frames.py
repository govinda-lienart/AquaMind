#IMPORTS
import cv2
import os
from datetime import datetime 
import yaml

# LOADING CONFIG FILE
with open('config.yaml') as f: 
    cfg = yaml.safe_load(f) # reads file and converts with load into python dictionary cfg 

# CONNECT WITH SQL

from db import get_connection
conn = get_connection()

# CREATE AN SQL EXECUTOR
cursor = conn.cursor() # executor - queries - tele operator - action part of the connect pipe

# FORMATING VIDEO AND TIME NAMING
# time_formating
format_now = datetime.now().strftime("%Y%m%d_%H%M")

# video_forma
video_path = cfg['extract_frames']['video_path']
video_name_ext = os.path.basename(video_path) 
video_name = os.path.splitext(video_name_ext)[0]

# time and video format
video_folder_name = f"frames_{video_name}_{format_now}"

# LOADING THE VIDEO
cap = cv2.VideoCapture(video_path) # loading the video

# CREATE FRAME DIRECTORY
os.makedirs(cfg['extract_frames']["frames_dir"], exist_ok=True) # creat dirctory frame in case it doesnt exist
os.makedirs(f"{cfg['extract_frames']['frames_dir']}/{video_folder_name}", exist_ok=True) # creat dirctory frame in case it doesnt exist

# EXTRACT FRAME PER SEC
fps = round(cap.get(cv2.CAP_PROP_FPS)) # fps prints 59.92 - not round - issue with modulo - therefore round.
print(f"Number of frames per second: {fps}")              
# LOOP TO EXTRACT FRAME AND STORE DATA IN SQL DB
frame_count = 0

while True:
    ret, frame = cap.read() #  ret (short for return) == boolean (True if info, False if none) # frame is the Numpy array - grid of pixels

    if not ret: # if returns false break
        break

    if frame_count % fps == 0:
        print (f"saving frame {frame_count}")
        frame_name = f"frame_{frame_count}_{video_name}_{format_now}"
        filename = f"{cfg['extract_frames']['frames_dir']}/{video_folder_name}/{frame_name}.png"
        cv2.imwrite(filename,frame)
        timestamp = frame_count / fps       
        cursor.execute(
            "INSERT INTO frames (video_path, frame_path, frame_number, timestamp, extracted_at) VALUES (%s, %s, %s, %s, %s)",
            (video_path, filename, frame_count, timestamp, datetime.now())
        )
    frame_count += 1

# SAVE SQL DB
conn.commit() # saves all the inserts to the database permanently

# CLOSURE OF THE PIPE
cap.release() # close video reading
conn.close()  # closes the connection to MySQL
print("Done")


