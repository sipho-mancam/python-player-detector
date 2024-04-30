from ultralytics import YOLO
import cv2 as cv
from pathlib import Path
import os
# Load a model
model = YOLO('./runs/detect/train/weights/best.pt')  # load a custom model
# model = YOLO('yolov8x.pt')
# Run batched inference on a list of images
# results = model(r"F:\Loftus  Night - 02-04-2024\Processed\2", stream=True)  # return a generator of Results objects

def getFilesList(dir:Path=""):
    srcDir = dir
    result = []
    for file in os.scandir(srcDir):
        result.append(file.path)
    return result

filesList = getFilesList(r"F:\Loftus  Night - 02-04-2024\Processed\3")
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# print(os.path.exists(Path(r"C:\Users\sipho-mancam\Videos\Video Systems Testing\Untitled 03.mp4").resolve().as_posix()))

# cap = cv.VideoCapture(Path(r"C:\Users\sipho-mancam\Videos\Video Systems Testing\Untitled 03.mp4").resolve().as_posix())

# frame = None
# while cap.isOpened():
#         if cap.isOpened():
#             _, frame = cap.read()
#             result = model.track(frame, persist=True)
#             annotated_i = result[0].plot()
#             cv.imshow("Preview", annotated_i)
#             cv.waitKey(1)

for file in filesList:
    if os.path.isfile(file):
        result = model.track(file, imgsz=1248, persist=True)
        # result2 = model.predict(file, imgsz=1248)
        annotated = result[0].plot()
        # for r in result:
        #     annotated = r.plot()
            # xyxys = r.boxes.xyxy
            # for xyxy in xyxys:
            #     cv.rectangle(r.orig_img, (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv.imshow("Preview", annotated)
        cv.waitKey(1)

# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk