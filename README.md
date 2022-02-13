Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. 

YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT.To Be Integrated With a Custom Action Classifier to predict the action performed by the detected basktball players

## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
Download our custom-basketball-dataset: https://drive.google.com/file/d/1vbW05VxQYuuvRs5GTdn5BGVnInr_o6Ts/view


Copy and paste **yolov4.weights** from your downloads folder into the **'data'** folder of this repository. Make sure to have the exact same name


## Running the Tracker with YOLOv4 Command (Without API Integration)
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.
```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)


## Running the Tracker with YOLOV4 using Endpoints
Navigate to the main directory and run the following command 

```bash
uvicorn main:app 
```
Running the application might take some while to load the model and expose the endpoints. You should be able to see something similar to

```bash
←[32mINFO←[0m:     Started server process [←[36m8500←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://127.0.0.1:8000←
```

Navigate to 

```bash
http://localhost:8000/docs
```
![image](https://user-images.githubusercontent.com/45897168/153769232-fe028940-2e03-4c2f-9514-483ca1ce794d.png)

You have set of endpoints:

1. Upload Videos
2. Check the videos that are uploaded (unprocessed)
3. Select an uploaded video to be processed
4. Download the video that was processed by the model

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [AI Guy](https://github.com/theAIGuysCodee)
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
