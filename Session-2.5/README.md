# **Assignment - 2.5 | PyTorch 101**

## Write a neural network that can:
    
    1. take 2 inputs:
        an image from the MNIST dataset (say 5), and
        a random number between 0 and 9, (say 7)

    2. and gives two outputs:
        the "number" that was represented by the MNIST image (predict 5), and
        the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

![Image](fig.png "Figure")

    3. you can mix fully connected layers and convolution layers
    4. you can use one-hot encoding to represent the random number input and the "summed" output.
        1. Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
        2. Sum (13) can be represented as:
            1. 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
            2. 0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010

## **Solution Approach** 

## 1. Data Representation

    Input:

        1. Mnist Data: 28X28 Image and it's label
        2. Random Number: One-Hot encoded (7 : 0 0 0 0 0 0 0 1 0 0)
    Output:

        1. Mnist Image Label
        2. Sum of two Numbers: One-Hot Ecoded (13: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0)
    
## 2.  Data Generation Strategy:

```python
class CreateDataSet(Dataset):
def __init__(self, mnist_data):
    self.data = mnist_data
    self.rand_num = random.randrange(9)

def getrandom(self, num, bas):
    b = np.zeros(bas)
    b[num] = 1
    return b


def __getitem__(self, index):
    num_image, actual_num = self.data[index]
    random_num = (index+actual_num+self.rand_num)%9
    random_number = self.getrandom(random_num, 10)
    actual_sum = random_num + actual_num
    actual_sum = self.getrandom(actual_sum, 19)
    
    return num_image, random_number, actual_num, actual_sum

def __len__(self):
    return len(self.data)
```
## 3. Strategy of Combining the two Inputs:

   **Random number data gets concateneted by the MNIST Image data once MNIST data gets Flattend**

```python
def forward(self, x, y):
    x = self.conv1(x)
    x = self.conv2(x)

    x = x.reshape(-1, 16*6*6)
    x1 = torch.cat([x, y], dim =1)
    
```

## 4. Way of evaluating the results:
```python
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def test(model, device, test_loader):
model.eval()
loss1 = 0
loss2 = 0
test_loss=0
correct_mnist_data = 0
correct_sum_predicted = 0

with torch.no_grad():
    for num_image, random_number, actual_mnist_num, actual_sum in test_loader:
        
        num_image, random_number, actual_mnist_num, actual_sum = num_image.to(device), random_number.float().to(device), actual_mnist_num.to(device), actual_sum.float().to(device)

        output1, output2 = model(num_image, random_number)
        loss1 += F.cross_entropy(output1, actual_mnist_num, reduction='sum').item()  # sum up batch loss
        loss2 += F.cross_entropy(output2, actual_sum, reduction='sum').item()

        test_loss = loss1+loss2

        a = get_num_correct(output1, actual_mnist_num)
        b = get_num_correct(output2, actual_sum.argmax(dim=1))

        correct_mnist_data += a
        correct_sum_predicted += b

test_loss /= len(test_loader.dataset)

print(f'Test Set: Total Loss:{test_loss:.3f}\
    MNIST Accuray:{100. * correct_mnist_data / len(test_loader.dataset):.3f}%\t\t \
    Sum Accuracy:{100. * correct_sum_predicted / len(test_loader.dataset):.3f}% \n')
```

## 5. Loss Function:

**Categorical Cross-Entropy**

## 6. Working CODE:
```python
import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import random
import numpy as np

mnist_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

class CreateDataSet(Dataset):
def __init__(self, mnist_data):
    self.data = mnist_data
    self.rand_num = random.randrange(9)

def getrandom(self, num, bas):
    b = np.zeros(bas)
    b[num] = 1
    return b


def __getitem__(self, index):
    num_image, actual_num = self.data[index]
    random_num = (index+actual_num+self.rand_num)%9
    random_number = self.getrandom(random_num, 10)
    actual_sum = random_num + actual_num
    actual_sum = self.getrandom(actual_sum, 19)
    
    return num_image, random_number, actual_num, actual_sum

def __len__(self):
    return len(self.data)

dataset = CreateDataSet(mnist_set)
train_set, test_set = torch.utils.data.random_split(dataset, [48000,12000])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=32)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3), nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1), # 26x26x16
                                nn.Conv2d(16,32,3), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout(0.1),# 24x24x32
                                nn.Conv2d(32,10,1), nn.ReLU(), #24x24x10
                                nn.MaxPool2d(2, 2), #12x12x16
            
                                )
        self.conv2 =  nn.Sequential(nn.Conv2d(10,16,3),nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1), #10x10x16
                                nn.Conv2d(16,16,3),nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1),   #8x8x16
                                nn.Conv2d(16,16,3),nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1),   #6x6x16
        )

        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out1 = nn.Linear(in_features=60, out_features=10)
        

        self.fc11 = nn.Linear(in_features=586, out_features=240)
        self.fc21 = nn.Linear(in_features=240, out_features=120)
        self.fc22 = nn.Linear(in_features=120, out_features=60)
        self.fc23 = nn.Linear(in_features=60, out_features=30)
        self.out2 = nn.Linear(in_features=30, out_features=19)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.reshape(-1, 16*6*6)
        #print(f'shape after flattening:{x.shape}')
        #print(f'shape of random number:{y.shape}')

        x1 = torch.cat([x, y], dim =1)
        #print(f'shape of x1:{x1.shape}')

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out1(x)

        x1 = F.relu(self.fc11(x1))
        x1 = F.relu(self.fc21(x1))
        x1 = F.relu(self.fc22(x1))
        x1 = F.relu(self.fc23(x1))
        x1 = self.out2(x1)

        x = F.softmax(x, dim=1)
        x1 = F.softmax(x1, dim=1)

        return x, x1
def get_num_correct(preds, labels):
return preds.argmax(dim=1).eq(labels).sum().item()

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (num_image, random_number, actual_mnist_num, actual_sum) in enumerate(pbar):
        #random_number = random_number.float()
        num_image, random_number, actual_mnist_num, actual_sum = num_image.to(device), random_number.float().to(device), actual_mnist_num.to(device), actual_sum.float().to(device)
        optimizer.zero_grad()
        output1, output2 = model(num_image, random_number)
        loss1 = F.cross_entropy(output1, actual_mnist_num)
        loss2 = F.cross_entropy(output2, actual_sum)
        
        total_loss = loss = loss1+ (loss2*4)
        total_loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss1={loss1.item():.4f} loss2={loss2.item():.4f} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    loss1 = 0
    loss2 = 0
    test_loss=0
    correct_mnist_data = 0
    correct_sum_predicted = 0

    with torch.no_grad():
        for num_image, random_number, actual_mnist_num, actual_sum in test_loader:
            
            num_image, random_number, actual_mnist_num, actual_sum = num_image.to(device), random_number.float().to(device), actual_mnist_num.to(device), actual_sum.float().to(device)

            output1, output2 = model(num_image, random_number)
            loss1 += F.cross_entropy(output1, actual_mnist_num, reduction='sum').item()  # sum up batch loss
            loss2 += F.cross_entropy(output2, actual_sum, reduction='sum').item()

            test_loss = loss1+loss2

            a = get_num_correct(output1, actual_mnist_num)
            b = get_num_correct(output2, actual_sum.argmax(dim=1))

            correct_mnist_data += a
            correct_sum_predicted += b

    test_loss /= len(test_loader.dataset)

    print(f'Test Set: Total Loss:{test_loss:.3f}\
    MNIST Accuray:{100. * correct_mnist_data / len(test_loader.dataset):.3f}%\t\t \
    Sum Accuracy:{100. * correct_sum_predicted / len(test_loader.dataset):.3f}% \n')

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 50):
    print("epoch %d"% epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```










