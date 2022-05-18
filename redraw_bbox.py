import cv2
import requests

from paths import video_received_dir, video_assigned_dir, api_url


def draw(video, bboxes, name):
    response = requests.get(api_url + 'videos/{}'.format(video))
    video_url = response.json()['videoRawUrl']
    vid = cv2.VideoCapture(video_url)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'H264')
    output_path = video_assigned_dir + '{}'.format(video)
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            out.release()
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        for box in bboxes:
            if box['frameNumber'] == frame_num and box['detectionType'] == 'person':
                frame = draw_bbox(frame, box, name)
                break
        out.write(frame)


def draw_bbox(image, boxes, player_name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pt1 = (int(boxes['x_min']), int(boxes['y_min']))
    pt2 = (int(boxes['x_max']), int(boxes['y_max']))
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
    cv2.rectangle(image, (pt1[0], pt1[1] - 30), (pt1[0] + len(player_name) * 14, pt1[1]), (255, 0, 0), -1)
    cv2.putText(image, player_name, (pt1[0], pt1[1] - 10), 0, 0.75, (0, 0, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return image
