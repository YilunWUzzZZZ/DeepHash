import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
from DataHandler import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_last = nn.MaxPool2d(2,2,padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(10, 16, (3, 3), padding=1)
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.fc_1 = nn.Linear(32 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 10)

    def forward(self, x):
        layer1_out = self.pool(self.relu(self.conv_1(x)))
        layer2_out = self.pool(self.relu(self.conv_2(layer1_out)))
        layer3_out = self.pool_last(self.relu(self.conv_3(layer2_out)))
        fc1_input = layer3_out.view(-1, 32 * 4 * 4)
        fc1_out = self.fc_1(fc1_input)
        out = self.fc_2(fc1_out)
        return out

mnistNet = MnistNet()
mnistNet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnistNet.parameters(),lr = 1e-3)

def train():
    checkCycle = 2000
    for epoch in range(2):
        total_loss = 0
        for i,(image, label) in enumerate(trainLoader):
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            out = mnistNet(image)
            loss = criterion(out, label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if(i%checkCycle == (checkCycle-1)):
                print("Ephoc %d, Running Loss : %.2f"%(epoch, total_loss/checkCycle))
                total_loss = 0

torch.save(mnistNet.state_dict(), 'C:/mnist/models/param.pkl')

def test():
    crt = 0
    total = 0
    for i, (image, label) in enumerate(trainLoader):
        image, label = image.to(device), label.to(device)
        out = mnistNet(image)
        _, predicted = torch.max(out, 1)
        crt += ((predicted == label).sum()).item()
        total += 4
    print("Accuracy : %.2f"%(float(crt)/total))

print("Start Training")
train()
print("Finished")
test()
