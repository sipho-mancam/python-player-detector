import cv2 as cv
import numpy as np

class BoxToPoint:
    def __init__(self, det_struct:dict, **kwargs)->None:
        self.__det_struct = det_struct # JSON representation of the detections object
        self.__xy = None # <x,y> coordinates of the object.
        self.__init()

    def __init(self):
        x1 = self.__det_struct.get('box').get("x1")
        y1 = self.__det_struct.get('box')["y1"]
        x2 = self.__det_struct.get('box')["x2"]
        y2 = self.__det_struct.get('box')["y2"]
        width = x2 - x1;
        self.__xy = (int(x2-(width/2)), int(y2))
        self.__det_struct["coordinates"] = self.__xy

    def get_struct(self):
        return self.__det_struct
    
    def draw_point(self, img)->cv.Mat:
        return cv.circle(img, self.__xy, 5, (255, 0, 0), thickness=cv.FILLED)
    
class CoordinatesFilter:
    def __init__(self, polygon:list, detections:dict)->None:
        self.__polygon = polygon
        self.__detections = detections

class BTransformations:
    def __init__(self):
        pass

    def transform(self, detections:list[dict])->list[dict]:
        return detections


class PerspectiveTransform(BTransformations):
    def __init__(self, src_poly:list, centre_pt:tuple[int, int])->None:
        super().__init__()
        self.__src_poly = src_poly
        self.__dst_poly = []
        self.__pers_matrix = None
        self.__centre_pt = centre_pt
        self.__sort_points()
        self.__calculate_dst_pts()

    def __calculate_dst_pts(self)->None:
        alpha = 250
        y3 = alpha + (2 * self.__centre_pt[1]) - self.__src_poly[0][1]
        x3 = self.__src_poly[0][0]
        x4 = x3 + (self.__src_poly[-1][0] - self.__src_poly[0][0])
        
        self.__dst_poly.append((x3, self.__src_poly[-1][1]+alpha))
        self.__dst_poly.append((x3, y3))
        self.__dst_poly.append((x4, y3))
        self.__dst_poly.append((x4, self.__src_poly[-1][1]+alpha))
        print("Dst Pts: ", self.__dst_poly)

    def __sort_points(self)->None:
        self.__src_poly = sorted(self.__src_poly)
        print("Sorted Pts: ", self.__src_poly)
        

    def transform(self, detections:list[dict])->list[dict]:
        # apply the perspective transform here..
        print(self.__dst_poly)
        return super().transform(detections)
    
    

class Transformer:
    def __init__(self, id=0)->None:
        self.__frame = None
        self.__detections_list = None
        self.__is_init = False
        self.__pitch_coordinates = []
        self.__window_name = "Transformer"
        self.__init_done = False
        self.__transformations = []
        self.__stream_id = id
        self.__centre_point = None
        self.__pers_transformer = None
        
    def init(self, img):
        self.__frame = img
        cv.namedWindow(self.__window_name, cv.WINDOW_NORMAL)
        # cv.displayOverlay(self.__window_name, "Select The Pitch Coordinates for the Mask")
        cv.setMouseCallback(self.__window_name, self.get_coordinat)
        cv.imshow(self.__window_name, img)

        print("Stream ID: ", self.__stream_id)

        while not self.__init_done:
            key = cv.waitKey(0)
            if self.__centre_point is not None:
                self.__pers_transformer = PerspectiveTransform(self.__pitch_coordinates, self.__centre_point)

            if key == 13:
                if len(self.__pitch_coordinates) >= 4:
                    if self.__stream_id == 1 and self.__centre_point is not None:
                        self.__init_done = True
                    elif self.__stream_id != 1:
                        self.__init_done = True

        

        self.__frame = cv.polylines(self.__frame, [np.array(self.__pitch_coordinates)], True, (0, 0, 255), 2)
        cv.imshow(self.__window_name, self.__frame)
        cv.waitKey(20)
        self.__is_init = True
        cv.destroyWindow(self.__window_name)

    def get_coordinat(self, event, x, y, flags, params)->None:
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.__pitch_coordinates) >= 4:
                self.__centre_point = (x , y)
            else:
                self.__pitch_coordinates.append((x,y))

            self.__frame = cv.circle(self.__frame, (x, y), 5, (255, 0, 0), thickness=cv.FILLED)
            self.__frame = cv.polylines(self.__frame, 
                                        [np.array(self.__pitch_coordinates)], 
                                        (not  (len(self.__pitch_coordinates) < 4 and len(self.__pitch_coordinates) > 0)), 
                                        (0,255, 0), 
                                        2)
                
            cv.imshow(self.__window_name, self.__frame)
        if event == cv.EVENT_MOUSEMOVE:
            if len(self.__pitch_coordinates) > 0 and len(self.__pitch_coordinates) <= 3:
                frame = self.__frame.copy()
                frame = cv.line(frame, self.__pitch_coordinates[-1], (x, y), (0,255, 0), 2)
                cv.imshow(self.__window_name, frame)

            
    def transform(self, img:cv.Mat, detections:list[dict])->tuple[list[dict], cv.Mat]:
        if not self.__is_init:
            self.init(img)
        
        img = cv.polylines(img, [np.array(self.__pitch_coordinates)], True, (255, 0, 0), 2)
        if self.__pers_transformer is not None:
            self.__pers_transformer.transform(detections)

        return img, detections
        
        # don some transformations here


def draw_points(img:cv.Mat, dets:list)->tuple[cv.Mat, dict]:
    a_img = img
    a_list = []
    for det in dets:
        bp = BoxToPoint(det)
        a_img = bp.draw_point(a_img)
        a_list.append(bp.get_struct())

    return (a_img, a_list)
        
