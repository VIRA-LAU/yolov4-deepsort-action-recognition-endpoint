import requests
import redraw_bbox
from paths import video_assigned_dir, api_url

timeout = 10000


# Firebase initialization


class assign_names:
    def __init__(self, bucket):
        self.bucket = bucket

    def assign(self):
        videos = requests.get(api_url + 'videos').json()
        video_id = None
        player_id = None
        for video in videos:
            if video['videoDetectUrl'] and not video['videoAssignedName']:
                video_id = video['videoId']
                video_name = video['videoName']
                print(video_id, video['videoName'])
                bbox_req = requests.get(api_url + 'object-detections/' + video_id).json()
                for bbox in bbox_req:
                    player_id = bbox['playerId']
                    if player_id:
                        player_name_req = requests.get(api_url + 'players/by/' + player_id).json()
                        player_name = player_name_req['firstName'] + ' ' + player_name_req['lastName']
                        sorted_frames = sorted(bbox_req, key=lambda d: d['frameNumber'])
                        redraw_bbox.draw(video_name, sorted_frames, player_name)
                        blob = self.bucket.blob('assigned_videos/' + video_name)
                        blob.upload_from_filename(video_assigned_dir + video_name, timeout=timeout)
                        blob.make_public()
                        print("Firebase URL:", blob.public_url)
                        assigned_url = {
                            "videoAssignedName": blob.public_url
                        }
                        r = requests.put(api_url + 'videos/update/assign/' + video_id, json=assigned_url)
                        break
