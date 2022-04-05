import requests

## Creating Initial Seeds
## Be consistent with the data


api_url = "http://localhost:5065/api/v1/"
defaultRoverId = '2db9de23-b477-45ad-b268-25229cb93ef4'
request = {
  "videoName": 'Sdr',
  "videoLocation": "LAU Court"
}
response = requests.post(api_url + 'videos/{}'.format(defaultRoverId), json=request)
print('video Id')
print(response.json()['videoId'])

videoIdReceived = response.json()['videoId']
response = requests.post(api_url + 'frames/{}'.format(videoIdReceived), json={})
print('frame Id')
print(response.json()['frameId'])

frameIdReceived = response.json()['frameId']


detection = {
  "detectionType": "Person",
  "detectionTrackingId": "1",
  "frameNumber": 5,
  "x_min": 300,
  "y_min": 250,
  "x_max": 150,
  "y_max": 12
}

response = requests.post(api_url + 'object-detections/{}'.format(frameIdReceived), json=detection)
print(response.text)