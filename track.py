import yolov5
import torch # to load the model
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
from datetime import datetime

# CONFIGURATION
# gathering path data
video_path = input("Please provide relative path of the video you want to process(e.g. videos/IMG_0350.MOV): ")
model_path = input ("Please provide relative path to model you will use (e.g. mlruns/743689458392478771/773d461e1e754e4d8b6f7d009b139b69/artifacts/weights/best.pt): ")

# generating output_video_path
video_path_basename_ext = os.path.basename(video_path) # IMG_0350.MOV
video_path_basename = os.path.splitext(video_path_basename_ext)[0] # takes first element of tuple ('IMG_0350', '.MOV') > IMG_0350
print(video_path_basename)
format_now = datetime.now().strftime("%Y%m%d_%H%M")
video_path_basename_time = f"tracked_{video_path_basename}_{format_now}" #tracked_IMG_0350_20260525_0922
output_video_path = f"output_video_deepsort/{video_path_basename_time}.mp4"  #output_video_deepsort/tracked_IMG_0350_20260525_0925.mp4
os.makedirs("output_video_deepsort", exist_ok=True)

# LOAD MODEL AND CREATE TRACKER
model = torch.hub.load('ultralytics/yolov5','custom', path=model_path) # custom means loading my own weight

tracker = DeepSort()

