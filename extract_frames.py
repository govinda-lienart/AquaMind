import cv2
import os

cap = cv2.VideoCapture("videos/IMG_9856.MOV") # loading the video
os.makedirs("frames", exist_ok=True) # creat dirctory frame in case it doesnt exist
fps = round(cap.get(cv2.CAP_PROP_FPS)) # fps prints 59.92 - not round - issue with modulo - therefore round.
print (fps) 


frame_count = 0

while True:
    ret, frame = cap.read() #  ret (short for return) == boolean (True if info, False if none) # frame is the Numpy array - grid of pixels

    if not ret: # if returns false break
        break

    if frame_count % fps == 0:
        print (f"saving frame {frame_count}")
        filename = f"frames/frame_{frame_count}.jpg"
        cv2.imwrite(filename,frame)

    frame_count += 1

cap.release() # close video reading
print("Done")


