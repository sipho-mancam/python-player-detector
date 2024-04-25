from ultralytics import YOLO
import cv2 as cv
import numpy as np



model = YOLO(r"./runs/detect/train/weights/best.pt")

def track(frame):
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    resized_image = np.resize(annotated_frame, (1920, 1080))
    print(results[0].tojson(normalize=True))
    cv.imshow("Tracking", resized_image)
    cv.waitKey(1)