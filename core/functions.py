import json
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from core.utils import read_class_names
from core.config import cfg
import requests

api_url = "https://stats-service-fyp-vira.herokuapp.com/api/v1/"


def send_request(data, allowed_classes, frame_num, frameId):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    # create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]


            print("str",xmin)
            if (class_name == 'madebasketball'):
                detection = {
                    "detectionType": class_name,
                    "detectionTrackingId": "0",
                    "frameNumber": frame_num,
                    "x_min": str(xmin),
                    "y_min": str(ymin),
                    "x_max": str(xmax),
                    "y_max": str(ymax)

                }
                print(detection)
                print("--------------",frameId)
                response = requests.post(api_url + 'object-detections/{}'.format(frameId),
                                         json=detection)
                print(response.text)
                print('----------------------', response.json()['objectDetectionId'], '----- ', response.status_code)


        else:
            continue
