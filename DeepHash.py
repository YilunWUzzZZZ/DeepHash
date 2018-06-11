import torch
import torch.nn as nn
import DataHandler
import torch.autograd
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MnistNet_Hash(nn.Module):
    def __init__(self, bits):
        super(MnistNet_Hash,self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_last = nn.MaxPool2d(2,2,padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(10, 16, (3, 3), padding=1)
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.fc_1 = nn.Linear(32 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, bits)

    def forward(self, x):
        layer1_out = self.pool(self.relu(self.conv_1(x)))
        layer2_out = self.pool(self.relu(self.conv_2(layer1_out)))
        layer3_out = self.pool_last(self.relu(self.conv_3(layer2_out)))
        fc1_input = layer3_out.view(-1, 32 * 4 * 4)
        fc1_out = self.fc_1(fc1_input)
        out = self.fc_2(fc1_out)
        return out

hashNet = MnistNet_Hash(24)
hashNet.to(device)

trainLoader = DataHandler.trainLoader
testLoader = DataHandler.testLoader

optimizer = optim.Adam(hashNet.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.4)
def lossFuc(b1,b2,y,m, bits, a):
    euclidean_dist = torch.pow((b1 - b2), 2)
    term1 = 0.5 * y * euclidean_dist
    term2 = 0.5 * (1 - y) * (m - euclidean_dist)[(m - euclidean_dist) > 0].sum()
    regular_term =a * ( torch.norm(torch.abs(b1) - torch.ones(bits), 1) + torch.norm(torch.abs(b2) - torch.ones(bits), 1) )
    return term1 + term2 + regular_term

def train_hash(bits, a):
    total_loss = 0
    batchsize = DataHandler.BatchSize
    pairs_perIter = batchsize * (batchsize-1)/2
    for e in range(3):
        scheduler.step()
        for i, (images, labels) in enumerate(trainLoader):
            optimizer.zero_grad()
            hashCodes = hashNet(images)
            loss = 0
            for i in range(batchsize):
                for  j in range(i+1, batchsize):
                    isSimilar = 0
                    if labels[i] == labels[j]:
                        isSimilar = 1
                        loss += lossFuc(hashCodes[i], hashCodes[j], isSimilar, 4*bits, a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if(i % 2000 == 1999):
                ave_loss = total_loss/ (2000 * pairs_perIter)
                total_loss = 0



