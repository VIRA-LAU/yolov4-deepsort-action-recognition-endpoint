import os
import requests
# comment out below line to enable tensorflow logging outputs
from core.functions import send_request
from paths import video_detected_dir, api_url

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet



class Process:

    def __init__(self, path, model, videoId):
        response = requests.get(api_url + 'videos/{}'.format(path))
        self.video_url = response.json()['videoRawUrl']
        self.path = path
        self.model = model
        self.videoId = videoId



    def detect(self):
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
        XYSCALE = cfg.YOLO.XYSCALE
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        input_size = 416

        infer = self.model.signatures['serving_default']

        vid = cv2.VideoCapture(self.video_url)
        out = None
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'H264')
        output_path = video_detected_dir+'{}'.format(self.path)
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

        frame_num = 0
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break


            #### For Each Frame Send Request to the save the frame

            CreateFrameResponse = requests.post(api_url + 'frames/{}'.format(self.videoId),json={})
            print('frame Id Received From Stats Service: ', CreateFrameResponse.json()['frameId'])
            print('---------------------')
            frame_num +=1
            print('Frame #: ', frame_num)
            print('---------------------')


            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()



            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)


            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)
            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            # custom allowed classes (uncomment line below to allow detections for only people)
            allowed_classes = ['person', 'basketball', 'madebasketball']
            frame = utils.draw_bbox(frame, pred_bbox, True)
            send_request(pred_bbox, allowed_classes, frame_num, CreateFrameResponse.json()['frameId'])
            #frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)





            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)



            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]


            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)





            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)


            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)


            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                #print(str(vid.get(cv2.CAP_PROP_POS_MSEC)))
                #print(class_name)
                #print(track.track_id)
                #print(bbox)
                xmin, ymin, xmax, ymax = bbox

                detection = {
                    "detectionType": class_name,
                    "detectionTrackingId": str(track.track_id),
                    "frameNumber": frame_num,
                    "x_min": xmin,
                    "y_min": ymin,
                    "x_max": xmax,
                    "y_max": ymax

                }

                if(class_name == 'person'):
                    print(detection)
                    response = requests.post(api_url + 'object-detections/{}'.format(CreateFrameResponse.json()['frameId']), json=detection)
                    print('----------------------', response.json()['objectDetectionId'], '----- ',response.status_code)




                 # draw bbox on screen
                if class_name == "person":
                    color = colors[int(track.track_id) % len(colors)]

                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                    # if enable info flag then print details about each track

                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #cv2.imshow('image', result)
            #cv2.waitKey(0)
            out.write(result)
