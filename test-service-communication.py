import requests

## Creating Initial Seeds
## Be consistent with the data


api_url = "http://localhost:5065/api/v1/"
createUser = {
  "firstName": "string",
  "lastName": "string",
  "email": "roland@example.com",
  "password": "string"
}


response = requests.post(api_url + 'register-user', json=createUser)
print('--------------Creating User Response--------------')
print(response.json())
print('--------------------------------------------------')
jsonResponse = response.json()
print(jsonResponse['userId'])


createVideo = {
  "videoName": "Shooting Session 3",
  "videoLocation": "LAU COURT",
  "userEmail": "roland@example.com"
}
response = requests.post(api_url + 'create-video', json=createVideo)
print('--------------Creating Video Response--------------')
print(response.json())
print('--------------------------------------------------')


createVideoSequence = {
  "videoName": "Shooting Session 3",
  "videoLocation": "LAU COURT",
  "userEmail": "roland@example.com",
  "videoSequenceName": "0"
}

response = requests.post(api_url + 'create-video-sequence', json=createVideoSequence)
videoSequenceId = response.json()['videoSequenceId']
print('--------------Creating Video Sequence Response--------------')
print(response.json())
print('--------------------------------------------------')


createDetection = {

  "detectionType": "string",
  "detectionTrackingId": "string",
  "actionOccurred": "string",
  "fps": 0,
  "videoSequenceId": videoSequenceId,
  "x_min": 0,
  "y_min": 0,
  "x_max": 0,
  "y_max": 0
}

response = requests.post(api_url + 'add-detection', json=createDetection)
print('--------------Creating Detection Sequence Response--------------')
print(response.json())
print('--------------------------------------------------')



