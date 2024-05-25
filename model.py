import torch.nn as nn
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
