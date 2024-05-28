import sqlite3
import cv2
import torch
import threading
import mediapipe as mp
from model import Model3
from my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify, request

app = Flask(__name__)

# Database setup
DATABASE = 'status.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT NOT NULL
            )
        ''')
        conn.commit()

def insert_status(state):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO status (state) VALUES (?)', (state,))
        conn.commit()

def get_next_status():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, state FROM status ORDER BY id LIMIT 1')
        row = cursor.fetchone()
        if row:
            cursor.execute('DELETE FROM status WHERE id = ?', (row[0],))
            conn.commit()
            return row[1]
        return None

def get_all_statuses():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, state FROM status ORDER BY id')
        rows = cursor.fetchall()
        return rows

# Video processing setup
#video_path = 'src/slide.mp4'
video_path = r"D:\2024\2024-1-1\Playground_app\app\src\main\res\raw\swing.mp4"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_path = r'D:\2023\2023_1_1\2023-RnE\save_by_loss\goodmodel3.pth'
model = Model3()
#model = torch.load(model_path)
model.eval()

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
def process_video():
    cap = cv2.VideoCapture(video_path)
    xy_list_list = []

    while True:
        ret, image = cap.read()
        if not ret:
            insert_status('Done')
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
                            insert_status('Backward!')
                        elif out == 1:
                            status = 'sit'
                            insert_status('Sit!')
                        elif out == 2:
                            status = 'slide'
                            insert_status('Slide!')
                        elif out == 3:
                            status = 'swing'
                            insert_status('Backward!')
                        elif out == 4:
                            status = 'walk'
                            insert_status('Walk!')



                    xy_list_list = []

    cap.release()

@app.route('/', methods=['GET'])
def get_status():
    status = get_next_status()
    if status:
        return jsonify({'state': status})
    return jsonify({'state': 'No Status'})

@app.route('/all', methods=['GET'])
def get_all():
    statuses = get_all_statuses()
    return jsonify(statuses)



if __name__ == '__main__':
    init_db()
    video_thread = threading.Thread(target=process_video)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)
