from ultralytics import YOLO
import cv2 as cv
import numpy as np
import json


model = YOLO(r"./runs/detect/train/weights/best.pt")

frame_count = 0

def track(frame):
    global frame_count
    results = model.track(frame, persist=True)
    with open(f'./tracking_data/frame '+str(frame_count)+".json", 'w') as fp:
        json.dump(json.loads(results[0].tojson(normalize=True)), fp)
        frame_count +=1
