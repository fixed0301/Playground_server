# 야매로 클라에 데이터 전송은 성공.
import torch
import mediapipe as mp
import cv2
import os
from model import Model3
from my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify, request
import threading

video_path = 'src/slide.mp4'
app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

status = {'state': 'Processing'}



cap = cv2.VideoCapture(video_path)
xy_list_list = []

while True:
    ret, image = cap.read()
    if not ret:
        status['state'] = 'Done'
        break

    image = cv2.resize(image, (400, 700))
    results = pose.process(image)

    if results.pose_landmarks:
        xy_list = []
        for x_and_y in results.pose_landmarks.landmark:
            xy_list.append(x_and_y.x)
            xy_list.append(x_and_y.y)
        xy_list_list.append(xy_list)

        if len(xy_list_list) == 20:
            status['state'] = 'Fun!'
        elif len(xy_list_list) == 30:
            status['state'] = 'Slide'
            xy_list_list = []
        print(status)

    else:
        pass

cap.release()


