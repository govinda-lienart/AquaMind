# IMPORTS

import os
import datetime
import mysql.connector

# ── DB CONNECTION ──────────────────────────────────────────────
conn = mysql.connector.connect( # the pipe - "phone line"
    host='localhost',
    port=3306,
    user='root',
    password='aquamind',
    database='aquamind'
)
cursor = conn.cursor()

# ── LOOP THROUGH LABEL FILES ───────────────────────────────────
labels_path = "annotations/annotations_IMG_9856_20260506_0957/labels" 
listing_labeltxt = os.listdir(labels_path)   # its a list: ['e6d83681-frame_360.txt', 'e6d83681-frame_420.txt', 'e6d83681-frame_480.txt', ...] 

for x in listing_labeltxt:   # e6d83681-frame_360.txt (labels) - one file at the time for each of the outer loop.

        #── PARSE FILENAME → FRAME NUMBER ─────────────────────────
        extract_frame_numb = x.split("-")[1] # frame_360.txt
        extracted_label_numb = int(os.path.splitext(extract_frame_numb)[0].split("_")[1]) # frame_360 then 360     


        # ── QUERY DB → GET FRAME ID ──────────────────────────────── 
        cursor.execute(                                                                                                                        
        "SELECT id FROM frames WHERE frame_number = %s", (extracted_label_numb,)   # gives id from the frames table where frame_number equals the frame number I extracted from the filename  
        )

        frame_id = cursor.fetchone()[0] # retrieves the first row, takes the first column (id) from for example frame number: 2520 → frame_id: (104,) with [0] it becomes frame_id: 104
        print()
        print(f"frame number: {extracted_label_numb} → frame_id: {frame_id}") #   frame number: 420 → frame_id: (69,)                                                                


        # ── READ ANNOTATION FILE ───────────────────────────────────
        annotation_txt = os.path.join(labels_path, x) # joining annotations/annotations_IMG_9856_20260506_0957/labels + e6d83681-frame_360.txt
        with open (annotation_txt, 'r') as f:
            lines = f.readlines() # the file has one line with elements and spaces 0 0.24 0.67....it will be converted into a list of one eleement  e.g  ['0 0.24 0.67 0.10 0.11\n']                                                     
        print(lines) # frame number: 1980 → frame_id: 95 ...['0 0.25006416837782325 0.5780287474332649 0.0958675564681725 0.08418891170431211\n']"""

        # ── EXTRACT BOUNDING BOX VALUES & INSERT INTO DB ───────────
        for value in lines: # will take each element in ['0 0.24 0.67 0.10 0.11\n']    the loop isn't for iterating over the 5 elements, it's for iterating over lines as maybe label text file has more than one fish.                          
            list_per_line = value.split()  # split() with no argument splits on any whitespace like spaces, tabs, newlines so that means "0 0.24 0.67 0.10 0.11\n".split() is gonna give ['0', '0.24', '0.67', '0.10', '0.11'] and the \n disappears automatically. 
            print(list_per_line) # ['0', '0.3262962012320328', '0.40657084188911713', '0.1258983572895277', '0.09445585215605753']
            class_id = int(list_per_line[0])                                                                  
            x_center = float(list_per_line[1])                                                                  
            y_center = float(list_per_line[2])                                                                  
            width = float(list_per_line[3])                                                                     
            height = float(list_per_line[4])
            label = "fish"
            created_at = datetime.datetime.now()
            
            cursor.execute(
            "INSERT INTO annotations (frame_id, class_id, label, x_center, y_center, width, height, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (frame_id, class_id, label, x_center, y_center, width, height, created_at)
            )

# ── COMMIT & CLOSE ─────────────────────────────────────────────
conn.commit()
conn.close()





