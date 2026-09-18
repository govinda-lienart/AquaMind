"""Registers each video from video_metadata.xlsx into the MySQL videos table.
   Usage: python scripts/sync_videos.py
"""

# IMPORTS
import logging
import os
from dotenv import load_dotenv
import mysql.connector
import openpyxl
import cv2
from scripts.console import banner, banner_sub

# CONSTANTS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# HELPER FUNCTIONS

def load_video_row(xlsx_path): # returns a list of dicts, one per video row
    """ loads an excel file and transform it into a dictionary """
    wb = openpyxl.load_workbook(xlsx_path) # creates a workbook object
    ws = wb["videos"] # grab the sheet videos
    headers = [cell.value for cell in ws[1]] # values
    rows = ws.iter_rows(min_row=2, values_only=True) # row min skip row 1 with headers # generator
    return [dict(zip(headers, row)) for row in rows] # returns a dictionary same as the loop version: zip each row against headers, wrap in dict, collect


def get_video_fps_resolution(file_path):
    """extract key video data from the video like FPS, width"""
    cap = cv2.VideoCapture(file_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS)) # FPS = 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1920
    if width >= 3840:
        resolution = '4K'
    elif width >= 1920:
        resolution = '1080p'
    else:
        resolution = '720p'
    cap.release()
    return fps, resolution





# MAIN

banner("SYNC VIDEOS")

banner_sub("Loading video metadata from xlsx")
video_rows = load_video_row("video_metadata.xlsx")
print(video_rows)

banner_sub("Reading fps/resolution for each video")
for row in video_rows:
    fps, resolution = get_video_fps_resolution(row["file_path"])
    print(row["file_path"], fps, resolution)

banner_sub("Reading parameters mysql")
load_dotenv() # loads .env data
conn = mysql.connector.connect(
    host = os.getenv('DB_HOST'),
    port = int(os.getenv("DB_PORT")),
    user = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    database = os.getenv("DB_NAME")
)
print(conn) # object

banner_sub("Reading video from mysql")
cursor = conn.cursor()
cursor.execute("SELECT * FROM videos")
rows = cursor.fetchall()
print(rows)
cursor.close()

banner_sub("inserting a test video")
row = video_rows[0] # download first video metadata
print(row) 
