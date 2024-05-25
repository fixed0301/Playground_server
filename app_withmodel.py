import torch
import mediapipe as mp
import cv2
import os

from model import Model3
from my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify, request

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_path = r'D:\2023\2023_1_1\2023-RnE\save_by_loss\goodmodel3.pth'
model = Model3()
model = torch.load(model_path)
model.eval()
video_path = 'src/slide.mp4'

app = Flask(__name__)

status = 'None'
status_2 = 'None'

chat = {
    'backward': '미끄럼틀을 역행하고 있습니다.',
    'sit': '앉아있습니다.',
    'slide': '미끄럼틀을 내려오고 있습니다.',
    'swing': '그네를 타고 있습니다.',
    'walk': '걷고 있습니다.',
    'collision': '충돌 위험이 있습니다'
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

@app.route('/', methods=['POST'])
def process_video_route(): # if only request, start model and return status
    global status, status_2
    if request:  # secondactivty 열렸을 때 초기에 request 한번만 전송되면
        cap = cv2.VideoCapture(video_path) # 얘가 직접 영상을 접근해서 실행한다.
        xy_list_list = []

        while True:
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.resize(image, (400, 700))
            results = pose.process(image)

            if results.pose_landmarks:
                xy_list = []
                for x_and_y in results.pose_landmarks.landmark:
                    xy_list.append(x_and_y.x)
                    xy_list.append(x_and_y.y)
                xy_list_list.append(xy_list)

                if len(xy_list_list) == 30:
                    dataset = [{'key': 0, 'value': xy_list_list}]
                    dataset = MyDataset(dataset)
                    dataset = DataLoader(dataset)

                    for data, label in dataset:
                        data = data.to(device)
                        with torch.no_grad():
                            result = model(data)
                            _, out = torch.max(result, 1)
                            out = out.item()

                            if out == 0:
                                status = 'backward'
                            elif out == 1:
                                status = 'swing'
                            elif out == 2:
                                status = 'slide'
                            elif out == 3:
                                status = 'swing'
                            elif out == 4:
                                status = 'walk'

                            if status != status_2:
                                status_2 = status
                                # 전송할 때만 JSON 반환
                                return jsonify({'status': status})

                    xy_list_list = []
            else:
                pass

        cap.release()
    # 상태가 변경되지 않았을 경우 빈 응답 반환
    return '', 204

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
