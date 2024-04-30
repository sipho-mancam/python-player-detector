import cv2 as cv
import numpy as np
from pathlib import Path
import os
from threading import Thread, Event
from pypylon import pylon
import cv2
import threading


class BInputSource:
    def __init__(self, i_type:str, stream_id):
        self.__type = i_type
        self.__stream_id = stream_id




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
    




class InputStreamB:
    def __init__(self, camera:pylon.InstantCamera):
        print(camera)
        self.camera = camera
        self.is_grabbing = False
        self.grab_thread = None
        self.frames_buffer = []
        self.converter = pylon.ImageFormatConverter()
        self.data_ready = Event()
        self.init()

    def init(self):
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def start_grabbing(self):
        self.camera.StartGrabbing()
        self.is_grabbing = self.camera.IsGrabbing()
        self.grab_thread = threading.Thread(target=self._grab_loop)
        self.grab_thread.start()

    def Is_grabbing(self)->bool:
        return self.camera.IsGrabbing()

    def _grab_loop(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                # Convert grabbed image to OpenCV format (assuming it's a Mono8 image)
                image = self.converter.Convert(grab_result)
                img = image.GetArray()
                self.frames_buffer.append(img)
                self.data_ready.set()
                if len(self.frames_buffer) > 10:
                    self.frames_buffer.pop(0) # remove the oldest frame
                cv2.namedWindow(self.camera.GetDeviceInfo().GetFriendlyName(), cv2.WINDOW_NORMAL)
                cv2.imshow(self.camera.GetDeviceInfo().GetFriendlyName(), img)
                cv.waitKey(1)
            grab_result.Release()

    def _record_video(self):
        pass

    def next(self)->cv.Mat:
        self.data_ready.wait()
        if len(self.frames_buffer) > 0:
            self.data_ready.clear()
            return self.frames_buffer.pop(0)
    

    def stop_grabbing(self):
        self.is_grabbing = False
        if self.grab_thread:
            self.grab_thread.join()

class DeviceFactory:
    def __init__(self):
        self.cameras = []

    def wait_for_cameras(self, num_cameras=3):
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        while len(devices) < num_cameras:
            # pass
            # Initialize cameras and add to the list
           devices = tlFactory.EnumerateDevices()
        
        for dev in devices:
            cam = pylon.InstantCamera()
            cam.Attach(tlFactory.CreateDevice(dev))
            self.cameras.append(cam)
        

    def get_input_stream(self, index=0):
        if index < len(self.cameras):
            return InputStreamB(self.cameras[index])
        else:
            raise IndexError("Camera index out of range")

if __name__ == "__main__":
    factory = DeviceFactory()
    factory.wait_for_cameras(3)
    
    # Assuming you want to work with the first camera
    cam_stream = factory.get_input_stream(0)
    cam_stream_2 = factory.get_input_stream(1)
    cam_stream_3 = factory.get_input_stream(2)

    cam_stream.start_grabbing()
    cam_stream_2.start_grabbing()
    cam_stream_3.start_grabbing()

    input("Press Enter to stop grabbing...")
    cam_stream.stop_grabbing()
    cv2.destroyAllWindows()
