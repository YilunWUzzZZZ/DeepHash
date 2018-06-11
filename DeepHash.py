import torch
import torch.nn as nn
import DataHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MnistNet_Hash(nn.Module):
    def __init__(self):
        super(MnistNet_Hash,self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_last = nn.MaxPool2d(2,2,padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(10, 16, (3, 3), padding=1)
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.fc_1 = nn.Linear(32 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 24)

    def forward(self, x):
        layer1_out = self.pool(self.relu(self.conv_1(x)))
        layer2_out = self.pool(self.relu(self.conv_2(layer1_out)))
        layer3_out = self.pool_last(self.relu(self.conv_3(layer2_out)))
        fc1_input = layer3_out.view(-1, 32 * 4 * 4)
        fc1_out = self.fc_1(fc1_input)
        out = self.fc_2(fc1_out)
        return out

hashNet = MnistNet_Hash()
hashNet.to(device)

trainLoader = DataHandler.trainLoader
testLoader = DataHandler.testLoader

def lossFuc(b1,b2,y,m, bits, a):
    euclidean_dist = torch.pow((b1 - b2), 2)
    term1 = 0.5 * y * euclidean_dist
    term2 = 0.5 * (1 - y) * (m - euclidean_dist)[(m - euclidean_dist) > 0].sum()
    regular_term =a * ( torch.norm(torch.abs(b1) - torch.ones(bits), 1) + torch.norm(torch.abs(b2) - torch.ones(bits), 1) )
    return term1 + term2 + regular_term

def train_hash()：

