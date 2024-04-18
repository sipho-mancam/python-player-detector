from input_pipeline import InputPipeline
from space_merger import SpaceMerger
from output_ import DetectionsOutput
from pathlib import Path
from utils import CThread
import cv2 as cv


class ProcessingPipeline:
    def __init__(self, stream1, stream2, stream3, weights)->None:
        self.__input_streams = [InputPipeline(stream1, 0, weights), 
                                InputPipeline(stream2, 1, weights),
                                InputPipeline(stream3, 2, weights)]

        self.__space_merger = SpaceMerger(self.__input_streams)
        self.__detections_output = None
        self.__stop = False

    def run(self)->None:
        # Run the input pipeline
        # run the space merger
        # run the output and repeat

        # initiailize the streams here
        for stream in self.__input_streams:
            stream.init() 
        
        try:
            cv.namedWindow("Preview 0", cv.WINDOW_NORMAL)
            cv.namedWindow("Preview 1", cv.WINDOW_NORMAL)
            cv.namedWindow("Preview 2", cv.WINDOW_NORMAL)
            # run the streams continuosly
            while not self.__stop:
                for stream in self.__input_streams:
                    stream.start()
                
                for idx, stream in enumerate(self.__input_streams):
                    res = stream.get_result()
                    if res is not None:
                        cv.imshow("Preview "+str(idx),res[0])
                    cv.waitKey(1)
        except KeyboardInterrupt as ke:
            for stream in self.__input_streams:
                stream.stop()
