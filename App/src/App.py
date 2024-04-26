# Python Detector Entry point
from input import InputStreams
from detector import Detector
from input_pipeline import InputPipeline
from coordinate_transforms import draw_points
import cv2 as cv
import json
from pathlib import Path
from processing_pipeline import ProcessingPipeline, load_mini_map, update_mini_map

def main():
    mini_map_window_name = "Mini-Map View"
    cv.namedWindow(mini_map_window_name, cv.WINDOW_NORMAL)

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