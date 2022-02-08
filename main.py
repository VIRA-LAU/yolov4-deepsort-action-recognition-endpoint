import shutil
from enum import Enum
import os
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse

app = FastAPI()


# uvicorn main:app --reload

class ModelName(str, Enum):
    yolov4 = "yolov4"



@app.get("/api/v1/public")
async def root():
    return {"message": "Working Public Endpoint"}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    isDarknetYOLOv4Available = os.path.isfile('./data/yolov4.weights')
    isTensorflowYOLOv4Available = os.path.isfile('./checkpoints/yolov4-416/saved_model.pb')
    if model_name == ModelName.yolov4:
        if not isDarknetYOLOv4Available:
            return {"model_name": model_name, "Darknet YOLOV4 Available": False}

        if not isTensorflowYOLOv4Available:
            return {"model_name": model_name, "Tensorflow YOLOV4 Available": False}

    return {"model_name": model_name, "Darknet YOLOV4 Available": True, "Tensorflow YOLOV4 Available": True, }



@app.post("/upload-video")
async def uploadVideo(file: UploadFile = File(...)):
    with open('./video_received/{}'.format(file.filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filenames": file.filename}


@app.get("/getVideo/{path}")
def getVideo(path):
    file_path = os.path.join("video_received/{}".format(path))
    if os.path.exists(file_path):
            return FileResponse(file_path, media_type="video/mp4", filename="{}.mp4".format(path))
    return {"error" : "File not found!"}






