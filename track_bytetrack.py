import os
import warnings
warnings.filterwarnings('ignore')
import cv2
import supervision as sv
from ultralytics import YOLO

# CONFIGURATION
input_video_path  = 'videos/IMG_0350.MOV'
model_path        = 'mlruns/743689458392478771/145ab81824c24d5da2ba0031d0de3d9b/artifacts/weights/best.pt'
output_video_path = 'output_video_bytetrack/IMG_0350_tracking1_2026-05-23_10-36-46.MOV'
max_seconds       = 10  # <-- I need adjust here to duration of the video

# LOAD MODEL
model = YOLO(model_path)

# CREATE TRACKER
tracker = sv.ByteTrack()

# OPEN VIDEO AND SET UP WRITER
os.makedirs('output_video_bytetrack', exist_ok=True)
cap       = cv2.VideoCapture(input_video_path)
width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps       = round(cap.get(cv2.CAP_PROP_FPS))
max_frames = max_seconds * fps
out       = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# MAIN LOOP
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break

    results = model(frame, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.class_id == 0]  # esnuring that only danio_rerio 
    tracked = tracker.update_with_detections(detections)

    for i in range(len(tracked)):
        x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
        track_id = tracked.tracker_id[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Fish {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}/{max_frames}  |  second {frame_count // fps}/{max_seconds}")

cap.release()
out.release()
print(f"Done. Saved to {output_video_path}")
