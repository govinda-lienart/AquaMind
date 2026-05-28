import mysql.connector
import os
import random
import shutil # shell utilities - resetting folder - clean slate
import yaml

from db import get_connection
conn = get_connection()

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# creation of sql executor that will select and fetch all frames that has annotation
created_at_input = cfg['prepare_dataset']['created_at']
video_path = cfg['prepare_dataset']['video_path']

reading_cursor = conn.cursor()

# GET VIDEO ID FROM VIDEOS TABLE
reading_cursor.execute("SELECT id FROM videos WHERE file_path = %s", (video_path,))
row = reading_cursor.fetchone()
if not row:
    print(f"Video {video_path} not registered. Run register_videos.py first.")
    conn.close()
    exit()
video_id = row[0]

reading_cursor.execute(
    'SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id WHERE f.video_id = %s AND a.created_at >= %s;',
    (video_id, created_at_input)
)
labeled_frames= reading_cursor.fetchall()
print()
print(f'raw tuple:\n\n{labeled_frames}')
print()

# shuffle all 
random.seed(42)
random.shuffle(labeled_frames)

print(f'shuffled tuple:\n\n{labeled_frames}')
print()

# Split 80/20 into train/val 
training_frames_80p = int(len(labeled_frames) * 0.8) # takes 80 percent of total frames
print(f'number of training frames:{training_frames_80p}')

select_training = labeled_frames[0:training_frames_80p] # selectiing 80 percent - so if 100 items - index 0 till 79 
select_validation = labeled_frames[training_frames_80p:] # selecting 20 percent then will go from index 80 till 99 (if 100 items)
print (f"number of validation frames: {len(select_validation)}")

# expected structure 

"""dataset/
└── dataset_IMG_0350_20260516_2148/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/"""

# create unique folder with videoname within dataset (eg. IMG_0350_20260516_2148)

extract_dir_name_tuple = labeled_frames[0][1] # selecting the first value typle as template anse select the path so [(579, 'frames/frames_IMG_0350_20260516_2148/frame_0_IMG_0350_20260516_2148.png') which is frames_IMG_0350_20260516_2148/frame_0_IMG_0350_20260516_2148.png
dir_basename = os.path.dirname(extract_dir_name_tuple) # extract parent folder name '
print(f'parent directory: {dir_basename}') 
video_basename_frames = (os.path.basename(dir_basename)) #'frames_IMG_0350_20260516_2148'
video_basename = video_basename_frames.replace("frames_", "", 1) # removed frames/ have cleaner video name IMG_0350_20260516_2148
print(f'video base name: {video_basename}')  # → 'IMG_0350_20260516_2148'

# resetting dataset folder to make sure file is clean when storing the symbiolinks
# creating directory for training and validation images..

experiment_name = created_at_input.replace(" ", "_").replace(":", "-") # adding the created_at info to the label of the runned yolo dataset...so i can run several times
if os.path.exists(f"dataset/{video_basename}_{experiment_name}"):
    shutil.rmtree(f"dataset/{video_basename}_{experiment_name}")

os.makedirs(f"dataset/{video_basename}_{experiment_name}/images/train/", exist_ok=True)
os.makedirs(f"dataset/{video_basename}_{experiment_name}/images/val/",exist_ok=True)
os.makedirs(f"dataset/{video_basename}_{experiment_name}/labels/train",exist_ok=True)
os.makedirs(f"dataset/{video_basename}_{experiment_name}/labels/val",exist_ok=True)


# FUNCTION TO GENERATE TRAINING AND VALIDATION DATASET

def generate_dataset(select_frames, split, video_basename, created_at_input, experiment_name, conn):# crearted at is added to avoid pulling out duplication of data from same video.
    reading_cursor = conn.cursor()
    for frame in select_frames:
        frame_id, frame_path = frame # extracting from the tuple only the path e.g 'frames/frames_IMG_9856_20260517_1221/frame_1080_IMG_9856_20260517_1221.png')
        frame_basename_png = os.path.basename(frame_path) # only "frame_1080_IMG_9856_20260517_1221.png"

        # storing the frame
        print(f'\n {split} Dataset -> Frame_id: {frame_id}')
        os.symlink(os.path.abspath(frame_path), f"dataset/{video_basename}_{experiment_name}/images/{split}/{frame_basename_png}") #create a symlink pointing from src(source-the real file that already exists) to dst (destination/where the symlink will appear) + note that symlinks should also have full patth to avoid issuers

        print(f"-Registering of frame symlink -> {video_basename}_{experiment_name}/images/{split}/{frame_basename_png} in dataset")

        # pulling out from sql the the annotation row for each frame_id 
        reading_cursor.execute(
        "SELECT class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s AND created_at >= %s",
        (frame_id, created_at_input))
        annotations = reading_cursor.fetchall()
        print(f"-SQL Annotation Fetching ->  {annotations}")

        # writing the annotation data that was pulled from sql into a text file with correct basename and stored in correct folder dataset
        frame_basename = os.path.splitext(frame_basename_png)[0] 
        frame_basename_txt = f"{frame_basename}.txt" # converted filename .png to filename.txt
        txt_file = f"dataset/{video_basename}_{experiment_name}/labels/{split}/{frame_basename_txt}"

        with open (txt_file, "w") as f:
            for row in annotations:
                    class_id, x_center, y_center, width, height = row # unpacking row
                    str_annot = f"{class_id} {x_center} {y_center} {width} {height}\n"
                    f.write(str_annot)
            print(f"-Storing annotation data in txt_file: {txt_file}")

generate_dataset(select_training, "train", video_basename, created_at_input, experiment_name, conn) # added video_basename to make the fucntion self contained, reusable and testable on itself
generate_dataset(select_validation, "val", video_basename, created_at_input, experiment_name, conn )

print(f"\n\033[92mDataset generated - check dataset/{video_basename}_{experiment_name}\033[0m")
