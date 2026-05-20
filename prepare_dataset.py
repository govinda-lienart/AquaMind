import mysql.connector
import os
import random
import shutil # shell utilities - resetting folder - clean slate

# Query MySQL for all labeled frame paths and their annotations

conn = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='aquamind',
    database='aquamind'
)

# creation of sql executor that will select and fetch all frames that has annotation
reading_cursor = conn.cursor()
reading_cursor.execute(
    'SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id WHERE video_path = "videos/IMG_0350.MOV";'
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

# create unique folder 

extract_dir_name_tuple = labeled_frames[0:training_frames_80p][0][1] # selecting the first value typle as template anse select the path so [(579, 'frames/frames_IMG_0350_20260516_2148/frame_0_IMG_0350_20260516_2148.png') which is frames_IMG_0350_20260516_2148/frame_0_IMG_0350_20260516_2148.png
dir_basename = os.path.dirname(extract_dir_name_tuple) # extract parent folder name '
print(f'parent directory: {dir_basename}') # 'frames/frames_IMG_0350_20260516_2148
video_basename = (os.path.basename(dir_basename))
print(f'video base name:s {video_basename}') # frames_IMG_0350_20260516_2148


# resetting dataset folder to make sure file is clean when storing the symbiolinks

if os.path.exists("dataset"):
    shutil.rmtree("dataset")

# creating directory for training and validation images..

os.makedirs(f"dataset/{video_basename}/images/train/", exist_ok=True)
os.makedirs(f"dataset/{video_basename}/images/val/",exist_ok=True)
os.makedirs(f"dataset/{video_basename}/labels/train",exist_ok=True)
os.makedirs(f"dataset/{video_basename}/labels/val",exist_ok=True)


#Create symlinks for training images (dataset/images/train/

for train_value in select_training:
    frame_id, frame_path = train_value # extracting from the tuple only the path e.g 'frames/frames_IMG_9856_20260517_1221/frame_1080_IMG_9856_20260517_1221.png')
    frame_basename_png = os.path.basename(frame_path) # only "frame_1080_IMG_9856_20260517_1221.png"

    print(f'\n\nFrame_id: {frame_id}')
    # storing the frame
    os.symlink(frame_path, f"dataset/{video_basename}/images/train/{frame_basename_png}") #create a symlink pointing from src(source-the real file that already exists) to dst (destination/where the symlink will appear)
    print(f"Registering of frame symlink -> {video_basename}/images/val/{frame_basename_png} in dataset")

    # pulling out from sql the the annotation row for each frame_id 

    reading_cursor = conn.cursor()
    reading_cursor.execute(
    "SELECT class_id, x_center, y_center, width, height FROM annotations WHERE frame_id = %s", (frame_id,)
    )
    annotations = reading_cursor.fetchall()
    print(f"SQL Annotation Fetching ->  {annotations}")

    # writing the annotation data that was pulled from sql into a text file with correct basename and stored in correct folder dataset
    frame_basename = os.path.splitext(frame_basename_png)[0] 
    frame_basename_txt = f"{frame_basename}.txt" # converted filename.jpg to filename.png
    txt_file = f"dataset/{video_basename}/labels/train/{frame_basename_txt}"


    with open (txt_file, "w") as f:
        for row in annotations:
            f.write(row)

    
#Create symlinks for validation images  (dataset/images/val/)

for val_value in select_validation:
    frame_id, frame_path = val_value
    frame_basename = os.path.basename(frame_path)
    sym = os.symlink(frame_path, f"dataset/{video_basename}/images/val/{frame_basename}") 




