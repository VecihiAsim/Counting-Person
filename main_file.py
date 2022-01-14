from __future__ import division, print_function, absolute_import
from timeit import time
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import warnings
import cv2
import numpy as np


class Counting():

    def __init__(self):
        self.yolo = YOLO()
        self.outputPath = "outputs/test_result2.avi"
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap =1.0
        self.peopleOut = 0
        self.peopleIn = 0
        self.model_filename = "model_data/mars-small128.pb"
        self.writeVideoFlag = True
        self.class_name = "person"
        self.logo = cv2.imread("photos/ayvos.png")
        self.logo_height, self.logo_width, _ = self.logo.shape
        self.polygonLine_coordinate = [(0, 0), (1280, 0), (1280, 360), (0, 90)]

    def people_count(self, inputPath):

        warnings.filterwarnings("ignore")
        encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric)
        tracker.polygonLine_coordinate = self.polygonLine_coordinate

        video_capture = cv2.VideoCapture(inputPath)

        if self.writeVideoFlag:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(self.outputPath, fourcc, 15, (w, h))
            list_file = open('outputs/ID_detections2.txt', 'w')
            frame_index = -1

        W = None
        H = None
        fps = 0.0

        while True:

            ret, frame = video_capture.read()
            frame[5:5 + self.logo_height, 5:5 + self.logo_width] = self.logo

            if ret != True:
                break
            t1 = time.time()

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            image = Image.fromarray(frame[..., ::-1])
            boxs = self.yolo.detect_image(image)
            features = encoder(frame, boxs)

            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections, H, W)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                # print("{} zone_query: ".format(track.track_id), self.zone_query(bbox))

                cmap = plt.get_cmap('tab20b')

                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(self.class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
                cv2.putText(frame, self.class_name + "/" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)

                if track.stateLocation == 1 and self.zone_query(bbox) == "zone2" and track.noConsider == False:
                    self.peopleOut += 1
                    track.stateLocation = 0
                    track.noConsider = True
                    cv2.line(frame, (360, 180), (1245, 360), (0, 255, 0), 2)

                if track.stateLocation == 0 and self.zone_query(bbox) == "zone1" and track.noConsider == False:
                    self.peopleIn += 1
                    track.stateLocation = 1
                    track.noConsider = True
                    cv2.line(frame, (360, 180), (1245, 360), (0, 255, 0), 2)

            cv2.line(frame, (360, 180), (1245, 360), (0, 0, 255), 2)

            info = [
                ("TotalUp", self.peopleIn),
                ("TotalDown", self.peopleOut)
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (5, 35 + ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Couinting People', frame)

            if self.writeVideoFlag:

                out.write(frame)
                frame_index = frame_index + 1
                list_file.write(str(frame_index) + ' ')
                if len(boxs) != 0:
                    for i in range(0, len(boxs)):
                        list_file.write(
                            str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
                list_file.write('\n')

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS= %f" % (fps))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        if self.writeVideoFlag:
            out.release()
            list_file.close()
        cv2.destroyAllWindows()

    def zone_query(self, bbox):

        bbox_center = self.center_bbox(bbox)
        point_bbox = Point(bbox_center)
        polygon = Polygon(self.polygonLine_coordinate)

        if polygon.contains(point_bbox) == True :
            return "zone1"

        else :
            return "zone2"

        return None

    def center_bbox(self, bbox):
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2

        return np.array([x, y])


run = Counting()
run.people_count("videos/test.mp4")