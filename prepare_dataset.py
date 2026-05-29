import mysql.connector
import os
import random
import shutil # shell utilities - resetting folder - clean slate
import yaml

from db import get_connection
conn = get_connection()

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

frames_folder = cfg['prepare_dataset']['frames_folder']

reading_cursor = conn.cursor()

frame_path_pattern = f"{frames_folder}%"
reading_cursor.execute(
    'SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id WHERE f.frame_path LIKE %s',
    (frame_path_pattern,)
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

# derive dataset folder name from frames_folder
video_basename = os.path.basename(frames_folder).replace("frames_", "", 1)  # 'IMG_0764_20260528_2116'
print(f'video base name: {video_basename}')

if os.path.exists(f"dataset/{video_basename}"):
    shutil.rmtree(f"dataset/{video_basename}")

os.makedirs(f"dataset/{video_basename}/images/train/", exist_ok=True)
os.makedirs(f"dataset/{video_basename}/images/val/", exist_ok=True)
os.makedirs(f"dataset/{video_basename}/labels/train", exist_ok=True)
os.makedirs(f"dataset/{video_basename}/labels/val", exist_ok=True)

# FUNCTION TO GENERATE TRAINING AND VALIDATION DATASET

def generate_dataset(select_frames, split, video_basename, conn):
    reading_cursor = conn.cursor()
    for frame in select_frames:
        frame_id, frame_path = frame
        frame_basename_png = os.path.basename(frame_path)

        print(f'\n {split} Dataset -> Frame_id: {frame_id}')
        os.symlink(os.path.abspath(frame_path), f"dataset/{video_basename}/images/{split}/{frame_basename_png}")
        print(f"-Registering of frame symlink -> {video_basename}/images/{split}/{frame_basename_png} in dataset")

        reading_cursor.execute(
            "SELECT class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s",
            (frame_id,))
        annotations = reading_cursor.fetchall()
        print(f"-SQL Annotation Fetching ->  {annotations}")

        frame_basename = os.path.splitext(frame_basename_png)[0]
        txt_file = f"dataset/{video_basename}/labels/{split}/{frame_basename}.txt"

        with open(txt_file, "w") as f:
            for row in annotations:
                class_id, x_center, y_center, width, height = row
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"-Storing annotation data in txt_file: {txt_file}")

generate_dataset(select_training, "train", video_basename, conn)
generate_dataset(select_validation, "val", video_basename, conn)

print(f"\n\033[92mDataset generated - check dataset/{video_basename}\033[0m")
