from ultralytics import YOLO
import cv2 as cv
from pathlib import Path
import os
# Load a model
model = YOLO('./runs/detect/train/weights/best.engine')  # load a custom model

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
for file in filesList:
    if os.path.isfile(file):
        result = model.predict([file, file, file], imgsz=1248)
        # result2 = model.predict(file, imgsz=1248)
        for r in result:
            xyxys = r.boxes.xyxy
            for xyxy in xyxys:
                cv.rectangle(r.orig_img, (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv.imshow("Preview", result[0].orig_img)
        cv.waitKey(1)

# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk