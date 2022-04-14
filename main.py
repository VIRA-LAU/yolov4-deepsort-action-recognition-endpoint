import json
import shutil
from enum import Enum
import os
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from os import walk
import requests
from object_tracker_endpoint import Process

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])

app = FastAPI()


# uvicorn main:app --reload
api_url = "https://stats-service-fyp-vira.herokuapp.com/api/v1/"


class ModelName(str, Enum):
    yolov4 = "yolov4"


class Person(str, Enum):
    email = ""

@app.get("/api/v1/public")
async def root():
    return {"message": "Working Public Endpoint"}


@app.get("/api/v1/get-videos")
async def root():
    response = requests.get(api_url + "videos")
    return {"message": response.json()}

@app.get("/api/v1/public/models/{model_name}")
async def get_model(model_name: ModelName ):
    isDarknetYOLOv4Available = os.path.isfile('./data/yolov4.weights')
    isTensorflowYOLOv4Available = os.path.isfile('./checkpoints/yolov4-416/saved_model.pb')
    if model_name == ModelName.yolov4:
        if not isDarknetYOLOv4Available:
            return {"model_name": model_name, "Darknet YOLOV4 Available": False}

        if not isTensorflowYOLOv4Available:
            return {"model_name": model_name, "Tensorflow YOLOV4 Available": False}

    return {"model_name": model_name, "Darknet YOLOV4 Available": True, "Tensorflow YOLOV4 Available": True, }


@app.post("/api/v1/public/upload-video")
async def uploadVideo(file: UploadFile = File(...)):
    with open('./video_received/{}'.format(file.filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    shutil.copy2('./video_received/{}'.format(file.filename), '../../fyp-interface/src/assets/raw-video/')

    # createVideoSequence = {
    #     "videoName": "Shooting Session 3",
    #     "videoLocation": "LAU COURT",
    #     "userEmail": email,
    #     "videoSequenceName": file.filename[0]
    # }
    #
    # response = requests.post(api_url + 'create-video-sequence', json=createVideoSequence)
    # if response.status_code != 409:
    #     return {"filenames": file.filename, "response-status": response.status_code, "response-message": response.json()}


    return {"filenames": file.filename}


@app.get("/api/v1/public/getVideo/{path}")
def getVideo(path):
    file_path = os.path.join("video_processed/{}".format(path))
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename="{}.mp4".format(path))
    return {"error": "File not found!"}


@app.get("/api/v1/public/processed-videos")
def getListOfProcessedVideos():
    filenames = next(walk('../../fyp-interface/src/assets/detected-and-tracked-video/'), (None, None, []))[2]
    return {"Processed Videos": filenames}


@app.get("/api/v1/public/received-videos")
def getListOfReceivedVideos():
    filenames = next(walk('../../fyp-interface/src/assets/raw-video'), (None, None, []))[2]
    return {"Received Videos": filenames}



@app.get("/api/v1/public/videos")
def getListOfAllVideoStatus():
    DetectedAndTracked = next(walk('../../fyp-interface/src/assets/detected-and-tracked-video/'), (None, None, []))[2]
    RawVideos = next(walk('../../fyp-interface/src/assets/raw-video'), (None, None, []))[2]
    Classified = next(walk('../../fyp-interface/src/assets/classified-video/'), (None, None, []))[2]

    listOfVideos = {
        "raw-videos": RawVideos,
        "detected-and-tracked": DetectedAndTracked,
        "classified": Classified
    }

    return listOfVideos

@app.get("/api/v1/public/process-video/{videoName}")
def ProcessVideo(videoName):
    defaultRoverId = '2d1fca36-c8b2-443b-b449-10dd2a3ca5a4'
    request = {
        "videoName": videoName,
        "videoLocation": "LAU Court"
    }
    response = requests.post(api_url + 'videos/{}'.format(defaultRoverId), json=request)
    videoId = response.json()['videoId']
    result = Process(videoName, saved_model_loaded, videoId);
    result.detect()
    return 'Success'


