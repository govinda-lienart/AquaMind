# IMPORTS

import os
import datetime
import mysql.connector

# CONNECTING WITH MYSQL DB AND CREATION OF EXECUTER

conn = mysql.connector.connect( # the pipe - "phone line"
    host='localhost',
    port=3306,
    user='root',
    password='aquamind',
    database='aquamind'
)
cursor = conn.cursor()

# LOOP THROUGH LABEL FILES, MATCH TO FRAME ID, READ BOUNDING BOX DATA                 
labels_path = "annotations/annotations_IMG_9856_20260506_0957/labels" 
listing_labeltxt = os.listdir(labels_path)

for x in listing_labeltxt:   # e6d83681-frame_360.txt (labels)

        # extract framenumber from file name in labels as an integer
        extract_frame_numb = x.split("-")[1] # frame_360.txt
        extracted_label_numb = int(os.path.splitext(extract_frame_numb)[0].split("_")[1]) # frame_360 then 360     


        # runs a query, which is stored inside cursor and saved  in frame_id  
        cursor.execute(                                                                                                                        
        "SELECT id FROM frames WHERE frame_number = %s", (extracted_label_numb,)   # gives id from the frames table where frame_number equals the frame number I extracted from the filename  
        )

        frame_id = cursor.fetchone()[0] # retrieves the first row, takes the first column (id) from for example frame number: 2520 → frame_id: (104,) with [0] it becomes frame_id: 104
        print(f"frame number: {extracted_label_numb} → frame_id: {frame_id}") #   frame number: 420 → frame_id: (69,)                                                                


        # extract info inside text file
        annotation_txt = os.path.join(labels_path, x) # joining annotations/annotations_IMG_9856_20260506_0957/labels + e6d83681-frame_360.txt
        with open (annotation_txt, 'r') as f:
            lines = f.readlines() # the file has one line with elements and spaces 0 0.24 0.67....it will be converted into a list of one eleement  e.g  ['0 0.24 0.67 0.10 0.11\n']                                                     
        print(lines)
        print() # frame number: 1980 → frame_id: 95 ...['0 0.25006416837782325 0.5780287474332649 0.0958675564681725 0.08418891170431211\n']"""

        for value in lines: # will take each element in ['0 0.24 0.67 0.10 0.11\n']                            
        values = value.split()  # split() with no argument splits on any whitespace like spaces, tabs, newlines so that means "0 0.24 0.67 0.10 0.11\n".split() is gonna give ['0', '0.24', '0.67', '0.10', '0.11'] and the \n disappears automatically. 
                



