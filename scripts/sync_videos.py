


import openpyxl
import cv2

def load_video_row(xlsx_path): # returns a list of dicts, one per video row
    """ loads an excel file and transform it into a dictionary """
    wb = openpyxl.load_workbook(xlsx_path) # creates a workbook object
    ws = wb["videos"] # grab the sheet videos
    headers = [cell.value for cell in ws[1]] # values
    rows = ws.iter_rows(min_row=2, values_only=True) # row min skip row 1 with headers # generator
    return [dict(zip(headers, row)) for row in rows] # returns a dictionary same as the loop version: zip each row against headers, wrap in dict, collect

# call function
print(load_video_row("video_metadata.xlsx"))

def get_video_fps_resolution(file_path):
    """extract key video data from the video like FPS, width"""
    cap = cv2.VideoCapture(file_path)
    print(cap.get(cv2.CAP_PROP_FPS)) # FPS = 60
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1920
    cap.release()


get_video_fps_resolution("output_fish_tracker/stage5_tracker_IMG_2349__ann_5r_8c_9r_10r_11c_2026_07_20_1107/stage5_tracker_IMG_2349__ann_5r_8c_9r_10r_11c_2026_07_20_1107.mp4") 
