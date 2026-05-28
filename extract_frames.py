#IMPORTS
import cv2
import os
from datetime import datetime 
import yaml

# LOADING SETTINGS 
with open('config.yaml') as f:  # it will allow to find out which video to process (video_path) and where to save the frames (frames_dir)
    cfg = yaml.safe_load(f) # reads file and converts with load into python dictionary cfg 

# CONNECT WITH MYSQL
from db import get_connection 
conn = get_connection() # reads already config file internally - the database section

# Safety — ensure unique constraint exists
cursor = conn.cursor() # executor # the contrains is same video can't have same frame_number....therefore should be unique(video_id, frame_number)

try: # just testing if the unique contrains was already added in the past onstraint doesn't exist → MySQL creates it → no error → continues ✓
    cursor.execute("ALTER TABLE frames ADD UNIQUE unique_video_frame (video_id, frame_number);")
except Exception as e: # if already added...error (means already added) - → constraint already exists → MySQL throws 1061 → caught → ignored → continues ✓
    if '1061' not in str(e): # str(e) convert the error in a string...so if 1016 not found = that means unique not established - crash the script
        raise  # ← re-throw the error, script crashes, you see the error
    print("Unique constraint check: already in place, skipping.")  # if 1016 found - so in str(e) then print unique command already established

# FORMATING VIDEO NAMING AND PATH
# time_formating
format_now = datetime.now().strftime("%Y%m%d_%H%M")

# video_forma
video_path = cfg['extract_frames']['video_path']
video_name_ext = os.path.basename(video_path) 
video_name = os.path.splitext(video_name_ext)[0]

# time and video format
video_folder_name = f"frames_{video_name}_{format_now}"

# GET VIDEO ID FROM VIDEOS TABLE
cursor.execute("SELECT id FROM videos WHERE file_path = %s", (video_path,))
row = cursor.fetchone()
if not row:
    print(f"Video {video_path} not registered. Run register_videos.py first.")
    conn.close()
    exit()  
video_id = row[0] # example video_id = 4 so if i have already registered a video_id 4 it will include this value when generating the new rows in frames with newly extracted frames.

# ── CHECK IF VIDEO ALREADY EXTRACTED ──────────────────────────
cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = %s", (video_id,))
if cursor.fetchone()[0] > 0:
    print(f"Frames for {video_path} already exist in the database. Delete them first if you want to re-extract.")
    conn.close()
    exit()

# LOADING THE VIDEO
cap = cv2.VideoCapture(video_path) # loading the video

# CREATE FRAME DIRECTORY
os.makedirs(cfg['extract_frames']["frames_dir"], exist_ok=True) #  which video to process
os.makedirs(f"{cfg['extract_frames']['frames_dir']}/{video_folder_name}", exist_ok=True)  # where to save frames

# EXTRACT FRAME PER SEC
fps = round(cap.get(cv2.CAP_PROP_FPS)) # fps prints 59.92 - not round - issue with modulo - therefore round.
print(f"Number of frames per second: {fps}")    
          
# LOOP TO EXTRACT FRAME AND STORE DATA IN SQL DB
frame_count = 0

while True:
    ret, frame = cap.read() #  ret (short for return) == boolean (True if info, False if none) # frame is the Numpy array - grid of pixels

    if not ret: # if returns false break
        break

    if frame_count % fps == 0: # selecting the 60th frame (so 1 per sec)
        print (f"saving frame {frame_count}")
        frame_name = f"frame_{frame_count}_{video_name}_{format_now}"
        filename = f"{cfg['extract_frames']['frames_dir']}/{video_folder_name}/{frame_name}.png"
        cv2.imwrite(filename,frame)
        timestamp = frame_count / fps       
        cursor.execute(
            "INSERT INTO frames (video_id, frame_path, frame_number, timestamp, extracted_at) VALUES (%s, %s, %s, %s, %s)",
            (video_id, filename, frame_count, timestamp, datetime.now())
        )
    frame_count += 1

# SAVE SQL DB
conn.commit() # saves all the inserts to the database permanently

# CLOSURE OF THE PIPE
cap.release() # close video reading
conn.close()  # closes the connection to MySQL
print(f"frames and sql-metadata were stored successfully with path {cfg['extract_frames']['frames_dir']}/{video_folder_name}")

