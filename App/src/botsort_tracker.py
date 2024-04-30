from ultralytics import YOLO
import cv2 as cv
import numpy as np
import json
from tracker.bot_sort import BoTSORT, STrack
from cfg.config_ import TrackingConf

model = YOLO(r"./runs/detect/train/weights/best.pt")
tracker = BoTSORT(TrackingConf(), 10)
frame_count = 0

def convert_to_output_results(arr):
    num_entries = len(arr)
    output_results = np.zeros((num_entries, 10))  # Assuming there are 10 elements in total

    for idx, entry in enumerate(arr):
        # Extract values from each dictionary entry
        x1, y1, x2, y2 = entry['box']['x1'], entry['box']['y1'], entry['box']['x2'], entry['box']['y2']
        confidence = entry['confidence']
        cls = entry['class']
        feature1, feature2 = entry['coordinates']
        # Assign values to the output_results array
        output_results[idx, :4] = [x1, y1, x2, y2]
        output_results[idx, 4] = confidence
        output_results[idx, 5] = cls
        output_results[idx, 6] = feature1
        output_results[idx, 7] = feature2
        # Assign placeholder values for features 3 to 9
        output_results[idx, 8:] = np.nan  # or any other placeholder value you prefer
    return output_results



def draw_bbox(frame, det)->cv.Mat:
    frame_c = frame
    x1 , y1, w, h = det._tlwh
    x2 = x1+w
    y2 = y1+h
    frame_c = cv.rectangle(frame_c, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    frame_c = cv.putText(frame_c, f"{det.track_id}", (int(x1), int(y1-2)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return frame_c

def draw_bbox_2(frame, entry)->cv.Mat:
    frame_c = frame
    x1, y1, x2, y2 = entry['box']['x1'], entry['box']['y1'], entry['box']['x2'], entry['box']['y2']
    frame_c = cv.rectangle(frame_c, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return frame_c

def track2(frame, detections):
    det_output = convert_to_output_results(detections)
    global tracker
    online_tracks = tracker.update(det_output, frame)
    tracking_results = {}
    res = []
    for i, t in enumerate(online_tracks):
        if t.is_activated:
            res.append({
                'coordinates':t.coordinates.tolist(),
                'tracking-id': t.track_id,
                'bbox': t._tlwh.tolist(),
                'conf': t.score
            })
            try:
                detections[i]['coordinates'] = t.coordinates
                detections[i]['track_id'] = t.track_id
            except IndexError as ie:
                pass
    tracking_results['tracks'] = res
    return detections, tracking_results
