from ultralytics import YOLO
import cv2 as cv
from threading import Thread, Event
from utils import CThread
import json
from coordinate_transforms import draw_points
import ultralytics

class ISDetector:
    def __init__(self, weights:str, engine:bool=False)->None:
        self.__weights = weights
        self.__engine = engine
        self.__detector = None
        self.__worker_thread = None
        self.__is_ready = Event()
        self.__init()

    def __init(self)->None:
        self.__detector = YOLO(self.__weights)
    
    def __worker(self, img:cv.Mat):
        self.__is_ready.clear()
        res = self.__detector.predict(img, imgsz=1248)
        self.__is_ready.set()
        return res
    

    def detect(self, img:cv.Mat):
        if self.__detector is None:
            self.__init()
        self.__worker_thread = CThread(target=self.__worker, args=(img,))
        self.__worker_thread.start()

    def get_result(self)->tuple[dict]|None:
        if self.__worker_thread is not None:
            self.__worker_thread.join()

            res = self.__worker_thread.get_return()
            if res is not None:
                clean_frame = res[0].orig_img.copy()
                res[0].orig_img = res[0].plot(line_width=2, labels=False)
                dets = json.loads(res[0].tojson())
                res[0].orig_img, s = draw_points(res[0].orig_img, dets)
            return res[0], s, clean_frame
        
    def is_ready(self): 
        return self.__is_ready.is_set()



class Detector:
    def __init__(self, weights:str, engine:bool=False)->None:
        self.__weights = weights
        self.__engine = engine
        self.__detector = None
        self.__detectors = []
        self.__init()

    def __init(self)->None:
        self.__detector = YOLO(self.__weights)
    
    def __worker(self, img:cv.Mat, det):
        detector = None
        if det is None:
            detector = YOLO(self.__weights)
        else:
            detector = det
        # print(det)
        return detector.predict(img, imgsz=1248)
    


    def detect(self, img_list:list[cv.Mat], parallel:bool=True):
        if self.__detector is None:
            self.__init()

        results = []
        if parallel:
            threads = []
            for idx, img in enumerate(img_list):
                if len(img_list) > len(self.__detectors):
                    self.__detectors.append(YOLO(self.__weights))
                
                t = CThread(target=self.__worker, args=(img, self.__detectors[idx], ))
                t.start()
                threads.append(t)

            for thread in threads:
                thread.join()
                res = thread.get_return()
                
                if res is not None:
                    res[0].orig_img = res[0].plot(line_width=2, labels=False)
                    results.append(res[0])
            return results


        
        if self.__engine:
            for img in img_list:
                res = self.__detector.predict(img, imgsz=1248)
                res[0].orig_img = res[0].plot(line_width=2, labels=False)
                results.append(res[0])
            return results
        else:
            res = self.__detector.predict(img_list, imgsz=1248)
            for r in res:
                r.orig_img = r.plot(line_width=2, labels=False)
            return res
        
