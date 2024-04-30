import cv2 as cv
import numpy as np
import os
import json
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
        self.__top_offset = 0
        self.__left_offset = 0
        self.__dst_width = 0
        self.__dst_height = 0
        self.__sort_points()
        self.__calculate_dst_poly()
        self.__init()
        

    def __init(self)->None:
        # get the transformation matrix
        self.__pers_matrix = cv.getPerspectiveTransform(np.array(self.__src_poly, dtype=np.float32), np.array(self.__dst_poly, dtype=np.float32))

    def __calculate_dst_poly(self):
        if self.__centre_pt is not None:
            self.__calculate_dst_pts()
        else:
            self.__calculate_dst_pts_wings()

    # calculates for the center cam --> Cam 2
    def __calculate_dst_pts(self)->None:
        alpha = 500
        y3 = alpha + (2 * self.__centre_pt[1]) - self.__src_poly[0][1]
        x3 = self.__src_poly[0][0]
        x4 = x3 + (self.__src_poly[-1][0] - self.__src_poly[0][0])
        
        self.__dst_poly.append((x3, self.__src_poly[-1][1]+alpha))
        self.__dst_poly.append((x3, y3))
        self.__dst_poly.append((x4, y3))
        self.__dst_poly.append((x4, self.__src_poly[-1][1]+alpha))

        x_vector = [x3, x4]
        y_vector = [y3, self.__src_poly[-1][1]+alpha]
        self.__left_offset = min(x_vector)
        self.__top_offset = min(y_vector)
        self.__dst_height = max(y_vector) - min(y_vector)
        self.__dst_width = max(x_vector) - min(x_vector)
     

    # calculates dst for cam 1, and cam 3 < side cams>
    def __calculate_dst_pts_wings(self):
        right_wing =  False
        y_vector = [y[1] for y in self.__src_poly]
        x_vector = [x[0] for x in self.__src_poly]
        alpha = 250
        x_0 = min(x_vector)    

         # determine if its the left wing or the right wing
        for point in self.__src_poly:
            if point[0] == x_0 and (point[1] - min(y_vector)) < 200:
                right_wing = True

        self.__left_offset = x_0 if x_0 > 0 else 1
        y_0 = min(y_vector) - alpha
        self.__top_offset = y_0 if y_0 > 0 else 1
        x_n = max(x_vector)
        y_n = max(y_vector) + alpha
        self.__dst_height = y_n-y_0
        self.__dst_width = x_n - x_0
        x_1 = x_n
        y_1 = y_0
        x_2 = x_0
        y_2 = y_n
        if not right_wing:
            self.__dst_poly.append((x_2, y_2))
            self.__dst_poly.append((x_0, y_0))
            self.__dst_poly.append((x_n, y_n)) 
            self.__dst_poly.append((x_1, y_1))
        else:
            self.__dst_poly.append((x_0, y_0))
            self.__dst_poly.append((x_2, y_2))
            self.__dst_poly.append((x_1, y_1)) 
            self.__dst_poly.append((x_n, y_n)) 
        

    def __sort_points(self)->None:
        self.__src_poly = sorted(self.__src_poly)
        # print("Sorted Pts: ", self.__src_poly)
        

    def transform(self, detections:list[dict])->list[dict]:
        super().transform(detections)

        src_pts = []
        for det in detections:
            if det.get('coordinates') is not None:
                src_pts.append(det.get("coordinates"))
        
        # apply the perspective transform here..
        trans = cv.perspectiveTransform(np.array(src_pts, dtype=np.float32)[None, :, :], self.__pers_matrix)
        result = []
        for transformed_point, det_ in zip(trans[0], detections):
            det_["coordinates"] = (int(transformed_point[0]), int(transformed_point[1]))
            if transformed_point[0] >= 0 and transformed_point[1] >=0:
                result.append((int(transformed_point[0]), int(transformed_point[1])))
        # print(self.__dst_poly)
        return detections, result
    
    def getDstPts(self)->list:
        if self.__centre_pt is not None:
            return self.__dst_poly
        else:
            dst_vector = []
            dst_vector.append(self.__dst_poly[0])
            dst_vector.append(self.__dst_poly[1])
            dst_vector.append(self.__dst_poly[3])
            dst_vector.append(self.__dst_poly[2])
            return dst_vector
        
    def get_offsets(self)->tuple:
       return self.__left_offset, self.__top_offset, self.__dst_width, self.__dst_height

       

colors = [[b for  b in range(256)], [g for  g in range(256)], [r for  r in range(256)]]
    
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
        self.__color = None
        self.b = 0
        self.g = 0
        self.r = 0

    def getDstPts(self):
        if self.__pers_transformer:
            return self.__pers_transformer.getDstPts()
        return []
    
    def write_calib(self):
        data = {}
        with open(f"{self.__stream_id}.json", "w") as fp:
            data["src_pts"] = self.__pitch_coordinates
            if self.__centre_point is not None:
                data["center_pt"] = self.__centre_point
            json.dump(data, fp)
    
    def read_calib(self):
        data = {}
        with open(f"{self.__stream_id}.json", 'r') as fp:
            data = json.load(fp)
            poly = data.get('src_pts')
            if poly is not None:
                self.__pitch_coordinates = poly
            if 'center_pt' in data:
                self.__centre_point = data['center_pt']
        return data
    
    def is_calib(self)->bool:
        return os.path.exists(f"{self.__stream_id}.json")
        
    def init(self, img):
        if self.is_calib():
            data = self.read_calib()
            self.__pers_transformer = PerspectiveTransform(self.__pitch_coordinates, self.__centre_point)
            return

        self.b = int(np.random.choice(np.array(colors[0])))
        self.g = int(np.random.choice(np.array(colors[1])))
        self.r = int(np.random.choice(np.array(colors[2])))
        self.__color = (self.b, self.g, self.r)
        self.__frame = img
        cv.namedWindow(self.__window_name, cv.WINDOW_NORMAL)
        # cv.displayOverlay(self.__window_name, "Select The Pitch Coordinates for the Mask")
        cv.setMouseCallback(self.__window_name, self.get_coordinate)
        cv.imshow(self.__window_name, img)

        while not self.__init_done:
            key = cv.waitKey(0)
            if self.__centre_point is not None:
                self.__pers_transformer = PerspectiveTransform(self.__pitch_coordinates, self.__centre_point)
            else:
                self.__pers_transformer = PerspectiveTransform(self.__pitch_coordinates, None)

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
        self.write_calib()
        
       
    def get_coordinate(self, event, x, y, flags, params)->None:
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

    def __normalize_coordinates(self, width, height, detections_t)->dict:
        detections_n = []
        offsets = self.__pers_transformer.get_offsets()

        for detection in detections_t:
            coord = detection.get('coordinates')
            x_n = ((coord[0] - offsets[0]) / offsets[2])
            y_n = ((coord[1]  - offsets[1])/ offsets[3])
            detection['coordinates'] = (x_n, y_n)
            detections_n.append(detection)
        return detections_n

            
    def transform(self, img:cv.Mat, detections:list[dict])->tuple[list[dict], cv.Mat]:
        if not self.__is_init:
            self.init(img)
        detections_t = detections
        img = cv.polylines(img, [np.array(self.__pitch_coordinates)], True, (255, 0, 0), 2)
       
        res_vector = None
        if self.__pers_transformer is not None:
            detections_t, res_vector = self.__pers_transformer.transform(detections)
            # detections_t = [{"coordinates": point, "color":(255, 0, 0)} for point in self.__pers_transformer.getDstPts()]
            # img  = cv.polylines(img, [np.array(self.__pers_transformer.getDstPts())], True, (255, 255, 255), 3)
            # print(detections_t)
            for point in detections_t:
                point["color"] = self.__color
            #     img = cv.circle(img, point['coordinates'], 15, (self.b, self.g, self.r), thickness=cv.FILLED)
            detections_t = self.__normalize_coordinates(img.shape[0], img.shape[1], detections_t)
        return img, detections_t, res_vector
        
        # don some transformations here


def draw_points(img:cv.Mat, dets:list)->tuple[cv.Mat, dict]:
    a_img = img
    a_list = []
    for det in dets:
        bp = BoxToPoint(det)
        a_img = bp.draw_point(a_img)
        a_list.append(bp.get_struct())

    return (a_img, a_list)
        
