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


def assign_id(detections, tl_coord)->int:
    ind = -1 
    minimum_distance = np.inf
    for idx, det in enumerate(detections):
        box = det['box']
        bbox = np.asarray([box['x1'], box['y1'], box['x2'], box['y2']], dtype=np.float64)
        bbox_c = STrack.tlbr_to_tlwh(bbox)
        min_dist = np.sqrt(np.sum((tl_coord - bbox_c)**2))
        if min_dist < minimum_distance:
            minimum_distance = float(min_dist)
            ind = idx
    return ind

def track2(frame, detections):
    det_output = convert_to_output_results(detections)
    global tracker
    online_tracks = tracker.update(det_output, frame)
    for t, det in zip(online_tracks, detections):
        # idx = assign_id(detections, t._tlwh)
        # if idx >=0:
        # det = detections[idx]
        det['track_id'] = t.track_id
        det['coordinates'] = t.coordinates
        
        # det = detections[idx]
        # det['track_id'] = t.track_id
        # det['coordinates'] = t.coordinates if t.coordinates is not None else det['coordinates']
    return detections

# def track(frame):
#     global frame_count

#     # tracked_frame = cv.resize(frame, (1920, 1080))
#     results = model.track(frame, persist=True)
#     annotated_frame = results[0].plot(line_width=2, labels=False)
#     track_ids = results[0].boxes.id.int().cpu().tolist()
#     boxes = results[0].boxes.xywh.cpu()

#     for box, track_id in zip(boxes, track_ids):
#         annotated_frame = cv.putText(annotated_frame, f"{track_id}", (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)

#     tracked_frame = cv.resize(annotated_frame, (1920, 1080))
#     cv.imshow("Tracking results", tracked_frame)
#     cv.waitKey(1)
#     # with open(f'./tracking_data/frame '+str(frame_count)+".json", 'w') as fp:
#     #     json.dump(json.loads(results[0].tojson(normalize=True)), fp)
#     #     frame_count +=1



