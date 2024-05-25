import cv2
import torch
import numpy as np
import mediapipe as mp
#import slack_sdk
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.recurrent_layer = nn.LSTM(hidden_size=100, input_size=66, bidirectional=True)
        self.nonLin = nn.BatchNorm1d(30)


        self.recurrent_layer2 = nn.LSTM(hidden_size=100, input_size=200, bidirectional=True) # biLSTM이라 input 2배로 늘림
        self.nonLin2 = nn.BatchNorm1d(30)

        self.conv = nn.Conv1d(30, 36, 7, 1)
        self.activation = nn.ReLU()  # or Leaky ReLU activation..?


        #self.dropout = nn.Dropout(0.5)
        self.classify_layer = nn.Linear(194, 5) # LSTM 출력 차원: 100, 두 번째 nn.BatchNorm1d 출력 차원: 35, nn.Conv1d 출력 차원: 36, : 100 + 35 + 36 = 171

        # # Weight initialization
        # init.xavier_uniform_(self.conv.weight)
        # init.xavier_uniform_(self.classify_layer.weight)

    def forward(self, input, h_t_1=None, c_t_1=None):
        rnn_outputs, (hn, cn) = self.recurrent_layer(input)
        lin1 = self.nonLin(rnn_outputs)

        rnn_outputs2, (hn2, cn2) = self.recurrent_layer2(lin1)

        lin2 = self.nonLin2(rnn_outputs2)
        conv = self.conv(lin2)
        activation = self.activation(conv)

        logits = self.classify_layer(activation[:,-1])
        return logits

model = Model3()
# 입력의 예제 텐서
sequence_length = 10  # 입력 시퀀스 길이
batch_size = 30  # 배치 크기
lstm_depth = 2  # LSTM 층의 깊이
model_dimension = 66  # 모델의 hidden state 차원

example_input = torch.randn(sequence_length,batch_size,model_dimension)
# 모델을 TorchScript로 변환
#traced_model = torch.jit.trace(model, example_input)
#traced_model.save("traced_model.pt")

# 저장된 모델 로드
loaded_model = torch.jit.load("traced_model.pt")

# 모델의 서명 확인
#print(loaded_model.code)

# torch.jit.script를 사용하여 모델을 스크립트 모드로 변환
scripted_model = torch.jit.script(loaded_model)
scripted_model.save("scripted_model.pt")

