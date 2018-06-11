import torchvision
import torchvision.transforms as transforms
import torch.utils.data

PATH = 'C:/mnist'

def dataNomalizer(path):
    dataset = torchvision.datasets.MNIST(path, transform=transforms.ToTensor())
    sum= 0
    cnt = 0
    for image, label in dataset:
        sum += image.sum()
        cnt += 1
    total = cnt * 28 * 28
    avg = sum/total
    sum = 0
    for image, label in dataset:
        sum += torch.pow((image - avg), 2).sum()
    dev = sum/total
    return avg, torch.sqrt(dev)


mean, std = dataNomalizer(PATH)

trainset = torchvision.datasets.MNIST(PATH, train = True, transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([mean], [std])
]))
testset = torchvision.datasets.MNIST(PATH, train = False, transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([mean], [std])
]))

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

