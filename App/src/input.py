import cv2 as cv
import numpy as np
from pathlib import Path
import os
from threading import Thread






# implement the input class
class InputStream:
    def __init__(self, path:Path, id, **kwargs)->None:
        self.__path = path
        self.__stream = [] # keeps list of 3 frame
        self.__files_list = []
        self.__current_index = 0
        self.__index = 0
        self.__worker = None
        self.__stream_id = 0
        self.__init()

    def __init(self)->None:
        if os.path.isdir(self.__path):
            for file in os.scandir(self.__path):
                if file.path.endswith(('.bmp', '.tiff', '.png', '.jpg', '.jpeg')):
                    self.__files_list.append(file.path)
        self.__bootstrap()
    
    def __bootstrap(self):
        for idx, file in enumerate(self.__files_list):
            self.__stream.append(cv.imread(file))
            self.__current_index += 1
            if self.__current_index >= 5:
                return

    def readOne(self):
        for i in range(1): 
            if len(self.__stream) > 10:
                return 
            if len(self.__files_list)-1 > self.__current_index:     
                self.__stream.append(cv.imread(self.__files_list[self.__current_index]))
                self.__current_index +=1

    
    def next(self)->cv.Mat:
        if (len(self.__stream)-1) < self.__index and self.__worker is not None:
            self.__worker.join()

        self.__worker = Thread(target=self.readOne).start()
        frame = self.__stream.pop(0)
        self.__index += 1
        return frame



class InputStreams:
    def __init__(self, num_of_streams, paths_list, **kwargs)->None:
        self.number_of_streams = num_of_streams
        self.paths_list = paths_list
        self.streams_list = []
        self.__init()

    def __init(self)->None:
        for idx, p in enumerate(self.paths_list):
            self.streams_list.append(InputStream(p,idx))
        
    def next(self)->list[cv.Mat]:
        result = []
        for stream in self.streams_list:
            frame = stream.next()
           
            if frame is not None:
                result.append(frame)
        return result