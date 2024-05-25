from torch.utils.data import Dataset
import torch
import numpy as np
class MyDataset(Dataset):
    def __init__(self, dataset): #모든 행동을 통합한 df가 들어가야함
        self.x = []
        self.y = []
        for dic in dataset:
            self.y.append(dic['key']) #key 값에는 actions 들어감
            self.x.append(dic['value']) #action마다의 data 들어감

    def __getitem__(self, index): #index는 행동의 index
        data = self.x[index] # x에는 꺼내 쓸 (행동마다 30개 묶음프레임)의 데이터
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))

    def __len__(self):  # 데이터셋의 길이 반환
        return len(self.x)