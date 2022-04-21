import subprocess

from paths import *

import json
import shutil
from enum import Enum
import os

import uvicorn
import firebase_admin
from firebase_admin import credentials, storage
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from os import walk
import requests

from assigning_names import assign_names
from object_tracker_endpoint import Process
from paths import video_received_dir, api_url, video_detected_dir, video_classified_dir

timeout = 500
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])

# Firebase initialization
cred = credentials.Certificate("./data/fire.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'fyp-interface.appspot.com'})
bucket = storage.bucket()

app = FastAPI()

# uvicorn main:app --reload


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
async def get_model(model_name: ModelName):
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
    output_file = video_received_dir + file.filename

    with open(video_received_dir + '{}'.format(file.filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not file.filename.lower().endswith('.mp4'):
        output_file = video_received_dir + os.path.splitext(file.filename)[0] + ".mp4"
        subprocess.call(['ffmpeg', "-hwaccel", "cuda", '-i', video_received_dir + file.filename, output_file])
        os.remove(video_received_dir + file.filename)
    # shutil.copy2('./video_received/{}'.format(file.filename), '../../fyp-interface/src/assets/raw-video/')

    blob = bucket.blob('raw_videos/' + os.path.split(output_file)[-1])
    blob.upload_from_filename(output_file, timeout=timeout)
    blob.make_public()
    print("Firebase URL:", blob.public_url)
    defaultRoverId = '2d1fca36-c8b2-443b-b449-10dd2a3ca5a4'
    request = {
        "videoName": os.path.split(output_file)[-1],
        "videoLocation": "LAU Court",
        "videoRawUrl": blob.public_url
    }
    print(request)
    response = requests.post(api_url + 'videos/{}'.format(defaultRoverId), json=request)

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

    return {"filenames": os.path.split(output_file)[-1]}


@app.get("/api/v1/public/getVideo/{path}")
def getVideo(path):
    file_path = os.path.join(video_detected_dir + "{}".format(path))
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename="{}.mp4".format(path))
    return {"error": "File not found!"}


@app.get("/api/v1/public/processed-videos")
def getListOfProcessedVideos():
    filenames = next(walk(video_detected_dir), (None, None, []))[2]
    return {"Processed Videos": filenames}


@app.get("/api/v1/public/received-videos")
def getListOfReceivedVideos():
    filenames = next(walk(video_received_dir), (None, None, []))[2]
    return {"Received Videos": filenames}


@app.get("/api/v1/public/videos")
def getListOfAllVideoStatus():
    DetectedAndTracked = next(walk(video_detected_dir), (None, None, []))[2]
    RawVideos = next(walk(video_received_dir), (None, None, []))[2]
    Classified = next(walk(video_classified_dir), (None, None, []))[2]

    listOfVideos = {
        "raw-videos": RawVideos,
        "detected-and-tracked": DetectedAndTracked,
        "classified": Classified
    }

    return listOfVideos


@app.get("/api/v1/public/process-video/{videoName}")
async def ProcessVideo(videoName):
    response = requests.get(api_url + 'videos/{}'.format(videoName))
    videoId = response.json()['videoId']
    result = Process(videoName, saved_model_loaded, videoId);
    result.detect()
    print('Detected')

    blob = bucket.blob('processed_videos/' + videoName)
    blob.upload_from_filename(video_detected_dir + videoName, timeout=timeout)
    blob.make_public()
    print("Firebase URL:", blob.public_url)

    detected_url = {
        "videoDetectUrl": blob.public_url
    }
    requests.put(api_url + 'videos/update/{}'.format(videoId), json=detected_url)

    return {'url': blob.public_url}


@app.get("/api/v1/public/getVideoUrl/{name}")
def getVideoUrl(name):
    try:
        video = requests.get(api_url + 'videos/{}'.format(name)).json()
        print(video['videoRawUrl'])
        return {'rawUrl': video['videoRawUrl']}
    except:
        return {"error": "File not found!"}


@app.get("/api/v1/public/processed-videosUrl")
def getListOfProcessedVideosUrl():
    videos = requests.get(api_url + 'videos').json()
    to_return = []
    for video in videos:
        if video['videoDetectUrl']:
            to_return.append(video['videoDetectUrl'])
    return {'processedVideos': to_return}


@app.get("/api/v1/public/received-videosUrl")
def getListOfReceivedVideosUrl():
    videos = requests.get(api_url + 'videos').json()
    to_return = []
    for video in videos:
        to_return.append(video['videoRawUrl'])
    return {'videosReceived': to_return}


@app.get("/api/v1/public/videosUrl")
def getListOfAllVideoStatusUrl():
    videos = requests.get(api_url + 'videos').json()
    DetectedAndTracked = []
    RawVideos = []
    Classified = []
    for video in videos:
        RawVideos.append(video['videoRawUrl'])
        if video['videoDetectUrl']:
            DetectedAndTracked.append(video['videoDetectUrl'])
        if video['videoClassifyUrl']:
            Classified.append(video['videoClassifyUrl'])
    listOfVideos = {
        "raw-videos": RawVideos,
        "detected-and-tracked": DetectedAndTracked,
        "classified": Classified
    }
    return listOfVideos


@app.get("/api/v1/public/process-videoUrl/{videoName}")
async def ProcessVideoUrl(videoName):
    try:
        response = requests.get(api_url + 'videos/{}'.format(videoName))
    except:
        return {"error": "File not found!"}
    videoId = response.json()['videoId']
    result = Process(videoName, saved_model_loaded, videoId);
    result.detect()
    print('Detected')

    blob = bucket.blob('processed_videos/' + videoName)
    blob.upload_from_filename(video_detected_dir + videoName, timeout=timeout)
    blob.make_public()
    print("Firebase URL:", blob.public_url)

    detected_url = {
        "videoDetectUrl": blob.public_url
    }
    requests.put(api_url + 'videos/update/{}'.format(videoId), json=detected_url)

    return {'url': blob.public_url}


@app.get("/api/v1/public/assign_names_to_videos")
def assign_names_to_videos():
    assigner = assign_names(bucket)
    assigner.assign()


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
