import requests

## Creating Initial Seeds
## Be consistent with the data


api_url = "http://localhost:5065/api/v1/"
# createUser = {
#   "firstName": "string",
#   "lastName": "string",
#   "email": "roland@example.com",
#   "password": "string"
# }
#
#
# response = requests.post(api_url + 'register-user', json=createUser)
# print('--------------Creating User Response--------------')
# print(response.json())
# print('--------------------------------------------------')
# jsonResponse = response.json()
# print(jsonResponse['userId'])
#
#
# createVideo = {
#   "videoName": "Shooting Session 3",
#   "videoLocation": "LAU COURT",
#   "userEmail": "roland@example.com"
# }
# response = requests.post(api_url + 'create-video', json=createVideo)
# print('--------------Creating Video Response--------------')
# print(response.json())
# print('--------------------------------------------------')
#
#
# createVideoSequence = {
#   "videoName": "Shooting Session 3",
#   "videoLocation": "LAU COURT",
#   "userEmail": "roland@example.com",
#   "videoSequenceName": "0"
# }
#
# response = requests.post(api_url + 'create-video-sequence', json=createVideoSequence)
# videoSequenceId = response.json()['videoSequenceId']
# print('--------------Creating Video Sequence Response--------------')
# print(response.json())
# print('--------------------------------------------------')


createDetection = {

  "videoName": "Shooting Session 3",
  "videoLocation": "LAU Court",
  "userEmail": 'roland@example.com',
  "videoSequenceName": 'shot_made_0.mp4'[0],
  "detectionType": 'class_name',
  "detectionTrackingId": '1',
  "FrameNumber":0,
  "fps": 1.3123123123123213,
  "x_min": 123,
  "y_min": 323,
  "x_max": 123,
  "y_max": 333

}

response = requests.post(api_url + 'add-detection', json=createDetection)
print('--------------Creating Detection Sequence Response--------------')
print(response.json())
print('--------------------------------------------------')



