import torch
import mediapipe as mp
import cv2
import os
from model import Model3
from my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify
import threading

video_path = 'src/slide.mp4'
app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

status = {'state': 'Processing'}

def process_video():
    global status

    cap = cv2.VideoCapture(video_path)
    xy_list_list = []

    while True:
        ret, image = cap.read()
        if not ret:
            status['state'] = 'Done'
            print(status)
            break  # 비디오가 끝나면 루프를 종료합니다.

        image = cv2.resize(image, (400, 700))
        results = pose.process(image)

        if results.pose_landmarks:
            xy_list = []
            for x_and_y in results.pose_landmarks.landmark:
                xy_list.append(x_and_y.x)
                xy_list.append(x_and_y.y)
            xy_list_list.append(xy_list)

            if len(xy_list_list) == 10:
                status['state'] = 'Fun!'
                print(status)
            elif len(xy_list_list) == 20:
                status['state'] = 'Slide'
                print(status)
                xy_list_list = []

        else:
            pass

    cap.release()

@app.route('/', methods=['GET'])
def get_status():
    return jsonify(status)

if __name__ == '__main__':
    video_thread = threading.Thread(target=process_video)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)
