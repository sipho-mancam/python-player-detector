# Python Detector Entry point
from input import InputStreams
from detector import Detector
from input_pipeline import InputPipeline
from coordinate_transforms import draw_points
import cv2 as cv
import json
from pathlib import Path
from processing_pipeline import ProcessingPipeline, load_mini_map, update_mini_map




# def main():
#     print("Boostraping Streams ...")
#     input_streams = InputStreams(3, [r'F:\Loftus  Night - 02-04-2024\Processed\3', 
#                                  r'F:\Loftus  Night - 02-04-2024\Processed\2', 
#                                  r'F:\Loftus  Night - 02-04-2024\Processed\1'])
    
#     print("Initializing the detector ...")
#     detector = Detector(r"./runs/detect/train/weights/best.engine", True)

#     cv.namedWindow("Preview 0", cv.WINDOW_NORMAL)
#     cv.namedWindow("Preview 1", cv.WINDOW_NORMAL)
#     cv.namedWindow("Preview 2", cv.WINDOW_NORMAL)
   

#     print("Running ...")
#     frames = input_streams.next()
#     while len(frames) > 0:
#         results = detector.detect(frames, True)
#         for index, result in enumerate(results):
#            dets = json.loads(result.tojson())
#            result.orig_img, s = draw_points(result.orig_img, dets)
#            cv.imshow("Preview "+str(index), result.orig_img)

#         cv.waitKey(1)
#         frames = input_streams.next()


def main():

    mini_map_window_name = "Mini-Map View"
    cv.namedWindow(mini_map_window_name, cv.WINDOW_NORMAL)
    # while True:
    #     mm_bg = load_mini_map(mini_map_window_name)
    #     update_mini_map(mini_map_window_name, mm_bg, [{"coordinates":(0,0)}])

    pp = ProcessingPipeline(
        Path(r"F:\Loftus  Night - 02-04-2024\Processed\3"),
        Path(r'F:\Loftus  Night - 02-04-2024\Processed\2'),
        Path(r'F:\Loftus  Night - 02-04-2024\Processed\1'),
        Path(r"./runs/detect/train/weights/best.engine")
    )
    pp.run(mini_map_window_name)

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ke:
        print(ke)