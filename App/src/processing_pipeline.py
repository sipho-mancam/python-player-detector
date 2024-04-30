from input import DeviceFactory, InputStreamB
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
        if coord is not None:
            x_scaled = x_offset + int(coord[0]*width)
            y_scaled = y_offset + int(coord[1]*height)
            clone_bg = cv.circle(clone_bg, (x_scaled, y_scaled), 20, det.get('color') if det.get('colors') else (255, 255, 0), cv.FILLED)
            if det.get('track_id') is not None:
                clone_bg = cv.putText(clone_bg, f"{det['track_id']}", (x_scaled-15, y_scaled+5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv.imshow(win_name, clone_bg)
    cv.waitKey(1)


class ProcessingPipeline:
    def __init__(self, stream1, stream2, stream3, weights)->None:
        self.__device__factory = DeviceFactory()
        self.__device__factory.wait_for_cameras(3)
        self.__input_streams = [InputPipeline(stream1, 0, weights), 
                                InputPipeline(stream2, 1, weights),
                                InputPipeline(stream3, 2, weights)]
        self.__space_merger = SpaceMerger(self.__input_streams)
        self.__detections_output = DetectionsOutput()
        self.__stop = False
        self.__frame_counter = 0

    def run(self, mm_win_name="")->None:
        # Run the input pipeline
        # run the space merger
        # run the output and repeat
        mm_bg = load_mini_map(mm_win_name)
        # initiailize the streams here
        for stream in self.__input_streams:
            stream.init() 
        try:
            for stream in self.__input_streams:
                stream.start()

            # run the streams continuosly
            while not self.__stop:
                cams_output = []
                cams_frames_output = []
                frames_clean = []
                for idx, stream in enumerate(self.__input_streams):
                    # wait for stream 1, 2 and then 3 
                    res = stream.get_result()
                    if res is not None:  
                        cams_frames_output.append(res[0])
                        cams_output.append(res[1])
                        frames_clean.append(res[-1])
                    
                if len(cams_output) == 3:
                    # Merge images clean for tracking
                    frame_track = self.__space_merger.merge_frame_for_tracking(frames_clean)
                    # Merge images dirty for preview 
                    merged_image = self.__space_merger.merge_frame(cams_frames_output)
                    # Merge detections results for mini_map <normalized>
                    merged_space = self.__space_merger.merge(cams_output)
                    cv.imshow("Preview WIndow", merged_image)
                    mini_map_data, structured_data = track2(frame_track, merged_space)

                    # Outputs
                    structured_data['frame_number'] = self.__frame_counter
                    self.__detections_output.update(structured_data) 
                    self.__detections_output.write_to_kafka()
                    update_mini_map(mm_win_name, mm_bg, mini_map_data)
                    self.__frame_counter +=1

        except KeyboardInterrupt as ke:
            for stream in self.__input_streams:
                stream.stop()




