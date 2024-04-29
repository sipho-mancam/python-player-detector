from input_pipeline import InputPipeline
from space_merger import SpaceMerger
from output_ import DetectionsOutput
from pathlib import Path
from utils import CThread
import cv2 as cv
import numpy as np



from botsort_tracker import *

def load_mini_map(win_name)->cv.Mat:
    mini_map_bg = cv.imread("./App/assets/Soccer_pitch_dimensions.png")
    ret = np.zeros(mini_map_bg.shape, dtype=np.uint8) + 128
    cv.imshow(win_name, ret)
    cv.waitKey(1)
    return mini_map_bg

def update_mini_map(win_name, bg_img, detections):
    width = 0.89 * bg_img.shape[1] 
    height = 0.895 * bg_img.shape[0]
    clone_bg = np.array(bg_img)
    x_offset = 140
    y_offset = 85
    for det in detections:
        coord = det['coordinates']
        x_scaled = x_offset + int(coord[0]*width)
        y_scaled = y_offset + int(coord[1]*height)
        clone_bg = cv.circle(clone_bg, (x_scaled, y_scaled), 20, det.get('color') if det.get('colors') else (255, 255, 0), cv.FILLED)
        if det.get('track_id') is not None:
            clone_bg = cv.putText(clone_bg, f"{det['track_id']}", (x_scaled-15, y_scaled+5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv.imshow(win_name, clone_bg)
    cv.waitKey(1)

class ProcessingPipeline:
    def __init__(self, stream1, stream2, stream3, weights)->None:
        self.__input_streams = [InputPipeline(stream1, 0, weights), 
                                InputPipeline(stream2, 1, weights),
                                InputPipeline(stream3, 2, weights)]

        self.__space_merger = SpaceMerger(self.__input_streams)
        self.__detections_output = None
        self.__stop = False

    def run(self, mm_win_name="")->None:
        # Run the input pipeline
        # run the space merger
        # run the output and repeat
        mm_bg = load_mini_map(mm_win_name)
        # initiailize the streams here
        for stream in self.__input_streams:
            stream.init() 
        try:
            # cv.namedWindow("Preview 0", cv.WINDOW_NORMAL)
            # cv.namedWindow("Preview 1", cv.WINDOW_NORMAL)
            # cv.namedWindow("Preview 2", cv.WINDOW_NORMAL)
            for stream in self.__input_streams:
                stream.start()
            # run the streams continuosly
            
            while not self.__stop:
                cams_output = []
                cams_frames_output = []
                for idx, stream in enumerate(self.__input_streams):
                    # wait for stream 1, 2 and then 3 
                    res = stream.get_result()
                    if res is not None:  
                        cams_frames_output.append(res[0])
                        cams_output.append(res[1])
                #         cv.imshow("Preview "+str(idx),res[0])
                # cv.waitKey(1)
                    
                if len(cams_output) == 3:
                    frame_track = self.__space_merger.merge_frame_for_tracking(cams_frames_output)
                    merged_space = self.__space_merger.merge(cams_output)
                    merged_image = self.__space_merger.merge_frame(cams_frames_output)
                    cv.imshow("Preview WIndow", merged_image)
                    
                    merged_space = track2(frame_track, merged_space)
                    update_mini_map(mm_win_name, mm_bg, merged_space)
                    # cams_output = []
                    
        except KeyboardInterrupt as ke:
            for stream in self.__input_streams:
                stream.stop()




