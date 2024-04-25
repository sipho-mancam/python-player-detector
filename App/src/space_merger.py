
from input_pipeline import InputPipeline
import numpy as np
import cv2 as cv


class SpaceMerger:
    def __init__(self, input_pipelines:list[InputPipeline])->None:
        self.__input_pipelines_list = input_pipelines
        self.__stream_results = [] # a list of tuples containing the current results 
        self.__left_wing = 7/20 
        self.__right_wing = 7/20
        self.__middle = 6/20
        self.__m_overlap = (1/20)*0.3

    # stream1 = 0-30%; stream2 = 31-70%; stream3 = 71-100%
    def merge(self, cams_detections:list[dict])->list[dict]:
        unified_space = []
        for idx, dets_group in enumerate(cams_detections):
           for det in dets_group:
               # left wing
                if idx == 0:
                    coord = det['coordinates']
                    x_scaled = coord[0] * self.__left_wing
                    det['coordinates'] = (x_scaled, coord[1])

                # middle 
                elif idx == 1:
                    coord = det['coordinates']
                    x_scaled = coord[0] * (self.__middle + 2*self.__m_overlap)
                    x_shifted = x_scaled + (self.__left_wing - self.__m_overlap)
                    det['coordinates'] = (x_shifted, coord[1])
                    
                # Right wing
                elif idx==2:
                    coord = det['coordinates']
                    x_scaled = coord[0] * self.__right_wing
                    x_shifted = x_scaled + (self.__left_wing + self.__middle)
                    det['coordinates'] =  (x_shifted, coord[1])
                    
                unified_space.append(det)
        return unified_space
    
    def merge_frame(self, frames_list:list)->cv.Mat:
        # create a tuple of the images in the list
        f_list_tuple = tuple(frames_list)
        merged_image = np.hstack(f_list_tuple, dtype=np.uint8)
        return merged_image


                   


