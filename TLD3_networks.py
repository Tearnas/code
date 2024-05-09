import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor Network """
    def __init__(self, state_num, action_num, hidden1=256, hidden2=256, hidden3=256, lstm_hidden=128, num_layers=1):
        super(ActorNet, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=state_num, hidden_size=lstm_hidden,num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, action_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _= self.lstm(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.sigmoid(self.fc4(x))
        #print(out.shape)

        return out

class CriticNet(nn.Module):
    """ Critic Network"""
    def __init__(self, state_num, action_num, hidden1=512, hidden2=512, hidden3=512):

        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_num, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x_1, a_1 = xa
        a_1 = a_1.squeeze(0)
        x = self.relu(self.fc1(x_1))
        x = self.relu(self.fc2(torch.cat([x, a_1], 1)))
        x = self.relu(self.fc3(x))
        out_1 = self.fc4(x)

        x_1, a_1 = xa
        a_1 = a_1.squeeze(0)
        x = self.relu(self.fc1(x_1))
        x = self.relu(self.fc2(torch.cat([x, a_1], 1)))
        x = self.relu(self.fc3(x))
        out_2 = self.fc4(x)

        return out_1, out_2
