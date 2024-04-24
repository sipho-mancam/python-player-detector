from input import InputStream
from detector import ISDetector
from coordinate_transforms import draw_points, Transformer
from pathlib import Path
import cv2 as cv
from utils import CThread
from threading import Event


class InputPipeline:
    def __init__(self, path:Path, id:int, weights:Path|str, engine:bool=True, **kwargs) -> None:
        self.__path = path
        self.__id = id;
        self.__input_stream = InputStream(self.__path, self.__id)
        self.__weights = weights
        self.__engine = engine
        self.__detector = ISDetector(self.__weights, self.__engine)
        self.__transformer = Transformer(id)
        self.__worker_thread = None
        self.__exit_event = Event()
        self.__result = None
        self.__is_result_ready = Event()
        self.__data_ready = Event()
        self.__started = False

    def init(self)->None:
        self.next()

    def getDstPts(self):
        return self.__transformer.getDstPts()


    def next(self)->tuple[dict, cv.Mat]:
        # read input from the input stream
        # Pass the input Matrix into the Detector
        # Get the results of the Detector
        # pass the results of the detector to the Transformer
        # populate the result list with the output of the transformer
        frame = self.__input_stream.next()
        self.__detector.detect(frame)
        res_obj, detections = self.__detector.get_result()
        if detections is not None and res_obj is not None:
            img, dets, res_vec  = self.__transformer.transform(res_obj.orig_img, detections)
           
            return img, dets, res_vec
        
    def __run(self)->None:
        while not self.__exit_event.is_set():
            self.__data_ready.clear()
            self.__result = self.next()
            self.__is_result_ready.set()
            self.__data_ready.wait()
            
         
    def get_stream_id(self)->int:
        return self.__id

    def isOutputReady(self)->bool:
        return self.__is_result_ready.is_set()
    
    def get_result(self)->tuple|None:
        self.__is_result_ready.wait() # wait for the data to be ready
        result = self.__result
        self.__is_result_ready.clear()
        self.__data_ready.set()
        return result


    def start(self)->None:
        if not self.__started:
            self.__worker_thread = CThread(target=self.__run)
            self.__worker_thread.start()
            self.__started = True
    
    def stop(self)->None:
        self.__exit_event.set()
        self.__worker_thread.join()
        del self.__worker_thread
