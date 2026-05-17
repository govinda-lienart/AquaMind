import mysql.connector
import os
import random

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
    'SELECT DISTINCT f.id, f.frame_path FROM annotations a JOIN frames f ON a.frame_id = f.id;'
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

training_frames_80p = int(len(labeled_frames) * 0.8) 
print(f'number of training frames:{training_frames_80p}')
print()

validation_frames_20p = int(len(labeled_frames) * 0.2)
print(f'number of training frames:{validation_frames_20p}')




# Split 80/20 into train/val



#Create symlinks for images



# Write .txt label files


