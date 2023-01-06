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
## 7. Training Logs:
```python
epoch 1
loss1=1.4714 loss2=2.9005 batch_id=749: 100%|██████████| 750/750 [00:15<00:00, 49.54it/s]
Test Set: Total Loss:4.403     MNIST Accuray:96.783%		      Sum Accuracy:11.117% 

epoch 2
loss1=1.4821 loss2=2.8687 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.82it/s]
Test Set: Total Loss:4.386     MNIST Accuray:97.883%		      Sum Accuracy:11.408% 

epoch 3
loss1=1.4920 loss2=2.9053 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.29it/s]
Test Set: Total Loss:4.380     MNIST Accuray:98.183%		      Sum Accuracy:11.608% 

epoch 4
loss1=1.4831 loss2=2.9152 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.36it/s]
Test Set: Total Loss:4.376     MNIST Accuray:98.325%		      Sum Accuracy:11.692% 

epoch 5
loss1=1.4941 loss2=2.8895 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.08it/s]
Test Set: Total Loss:4.382     MNIST Accuray:97.292%		      Sum Accuracy:12.858% 

epoch 6
loss1=1.4884 loss2=2.8228 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.92it/s]
Test Set: Total Loss:4.335     MNIST Accuray:97.567%		      Sum Accuracy:18.892% 

epoch 7
loss1=1.4864 loss2=2.6295 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.34it/s]
Test Set: Total Loss:4.214     MNIST Accuray:98.375%		      Sum Accuracy:30.025% 

epoch 8
loss1=1.4848 loss2=2.6382 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.26it/s]
Test Set: Total Loss:4.202     MNIST Accuray:98.642%		      Sum Accuracy:30.483% 

epoch 9
loss1=1.4923 loss2=2.6978 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 50.09it/s]
Test Set: Total Loss:4.185     MNIST Accuray:98.650%		      Sum Accuracy:31.767% 

epoch 10
loss1=1.4806 loss2=2.6665 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 55.04it/s]
Test Set: Total Loss:4.181     MNIST Accuray:98.775%		      Sum Accuracy:31.875% 

epoch 11
loss1=1.5009 loss2=2.6664 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.91it/s]
Test Set: Total Loss:4.107     MNIST Accuray:98.792%		      Sum Accuracy:40.858% 

epoch 12
loss1=1.4630 loss2=2.6215 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.44it/s]
Test Set: Total Loss:4.078     MNIST Accuray:98.758%		      Sum Accuracy:42.333% 

epoch 13
loss1=1.4776 loss2=2.6010 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 51.55it/s]
Test Set: Total Loss:4.074     MNIST Accuray:98.767%		      Sum Accuracy:42.458% 

epoch 14
loss1=1.4639 loss2=2.3858 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 52.90it/s]
Test Set: Total Loss:3.904     MNIST Accuray:98.592%		      Sum Accuracy:60.717% 

epoch 15
loss1=1.4703 loss2=2.4029 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.93it/s]
Test Set: Total Loss:3.880     MNIST Accuray:98.717%		      Sum Accuracy:62.700% 

epoch 16
loss1=1.4623 loss2=2.4158 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.34it/s]
Test Set: Total Loss:3.877     MNIST Accuray:98.567%		      Sum Accuracy:62.958% 

epoch 17
loss1=1.4935 loss2=2.4686 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.28it/s]
Test Set: Total Loss:3.873     MNIST Accuray:98.858%		      Sum Accuracy:63.125% 

epoch 18
loss1=1.5037 loss2=2.3673 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.45it/s]
Test Set: Total Loss:3.870     MNIST Accuray:98.792%		      Sum Accuracy:63.217% 

epoch 19
loss1=1.4679 loss2=2.3322 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.57it/s]
Test Set: Total Loss:3.868     MNIST Accuray:98.892%		      Sum Accuracy:63.400% 

epoch 20
loss1=1.4732 loss2=2.4481 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 55.10it/s]
Test Set: Total Loss:3.868     MNIST Accuray:98.783%		      Sum Accuracy:63.375% 

epoch 21
loss1=1.5259 loss2=2.4212 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.57it/s]
Test Set: Total Loss:3.863     MNIST Accuray:98.908%		      Sum Accuracy:63.600% 

epoch 22
loss1=1.4904 loss2=2.3124 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.58it/s]
Test Set: Total Loss:3.827     MNIST Accuray:98.600%		      Sum Accuracy:67.833% 

epoch 23
loss1=1.4704 loss2=2.4174 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.75it/s]
Test Set: Total Loss:3.811     MNIST Accuray:98.950%		      Sum Accuracy:68.900% 

epoch 24
loss1=1.4939 loss2=2.4482 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.93it/s]
Test Set: Total Loss:3.816     MNIST Accuray:98.775%		      Sum Accuracy:68.517% 

epoch 25
loss1=1.4771 loss2=2.3382 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.41it/s]
Test Set: Total Loss:3.807     MNIST Accuray:99.025%		      Sum Accuracy:69.167% 

epoch 26
loss1=1.4669 loss2=2.3242 batch_id=749: 100%|██████████| 750/750 [00:15<00:00, 49.07it/s]
Test Set: Total Loss:3.808     MNIST Accuray:98.883%		      Sum Accuracy:69.083% 

epoch 27
loss1=1.4617 loss2=2.3082 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.01it/s]
Test Set: Total Loss:3.807     MNIST Accuray:99.033%		      Sum Accuracy:69.158% 

epoch 28
loss1=1.4716 loss2=2.2314 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.10it/s]
Test Set: Total Loss:3.807     MNIST Accuray:98.908%		      Sum Accuracy:69.192% 

epoch 29
loss1=1.4768 loss2=2.3367 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.87it/s]
Test Set: Total Loss:3.805     MNIST Accuray:99.067%		      Sum Accuracy:69.208% 

epoch 30
loss1=1.4738 loss2=2.4448 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.17it/s]
Test Set: Total Loss:3.802     MNIST Accuray:99.050%		      Sum Accuracy:69.158% 

epoch 31
loss1=1.4614 loss2=2.3383 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.50it/s]
Test Set: Total Loss:3.772     MNIST Accuray:99.042%		      Sum Accuracy:72.608% 

epoch 32
loss1=1.4885 loss2=2.2652 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.02it/s]
Test Set: Total Loss:3.772     MNIST Accuray:99.133%		      Sum Accuracy:72.550% 

epoch 33
loss1=1.5028 loss2=2.2927 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.87it/s]
Test Set: Total Loss:3.775     MNIST Accuray:98.867%		      Sum Accuracy:72.442% 

epoch 34
loss1=1.4614 loss2=2.1847 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.28it/s]
Test Set: Total Loss:3.770     MNIST Accuray:99.142%		      Sum Accuracy:72.650% 

epoch 35
loss1=1.4619 loss2=2.3218 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.86it/s]
Test Set: Total Loss:3.769     MNIST Accuray:99.183%		      Sum Accuracy:72.658% 

epoch 36
loss1=1.4617 loss2=2.2449 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.73it/s]
Test Set: Total Loss:3.767     MNIST Accuray:99.200%		      Sum Accuracy:72.758% 

epoch 37
loss1=1.4711 loss2=2.2908 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.09it/s]
Test Set: Total Loss:3.769     MNIST Accuray:99.183%		      Sum Accuracy:72.725% 

epoch 38
loss1=1.5150 loss2=2.2972 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.93it/s]
Test Set: Total Loss:3.771     MNIST Accuray:99.150%		      Sum Accuracy:72.508% 

epoch 39
loss1=1.4612 loss2=2.2116 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.70it/s]
Test Set: Total Loss:3.770     MNIST Accuray:99.108%		      Sum Accuracy:72.675% 

epoch 40
loss1=1.4810 loss2=2.3127 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.45it/s]
Test Set: Total Loss:3.770     MNIST Accuray:99.058%		      Sum Accuracy:72.717% 

epoch 41
loss1=1.4814 loss2=2.2917 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.74it/s]
Test Set: Total Loss:3.770     MNIST Accuray:98.983%		      Sum Accuracy:72.633% 

epoch 42
loss1=1.4646 loss2=2.2595 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.71it/s]
Test Set: Total Loss:3.766     MNIST Accuray:99.242%		      Sum Accuracy:72.858% 

epoch 43
loss1=1.4613 loss2=2.3628 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 53.53it/s]
Test Set: Total Loss:3.762     MNIST Accuray:99.192%		      Sum Accuracy:73.617% 

epoch 44
loss1=1.4762 loss2=2.1943 batch_id=749: 100%|██████████| 750/750 [00:15<00:00, 48.24it/s]
Test Set: Total Loss:3.758     MNIST Accuray:99.158%		      Sum Accuracy:73.742% 

epoch 45
loss1=1.4612 loss2=2.2750 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 53.59it/s]
Test Set: Total Loss:3.761     MNIST Accuray:99.067%		      Sum Accuracy:73.592% 

epoch 46
loss1=1.4642 loss2=2.3062 batch_id=749: 100%|██████████| 750/750 [00:13<00:00, 54.04it/s]
Test Set: Total Loss:3.760     MNIST Accuray:99.192%		      Sum Accuracy:73.558% 

epoch 47
loss1=1.4772 loss2=2.2571 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 52.92it/s]
Test Set: Total Loss:3.759     MNIST Accuray:99.100%		      Sum Accuracy:73.633% 

epoch 48
loss1=1.4767 loss2=2.2449 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 51.94it/s]
Test Set: Total Loss:3.757     MNIST Accuray:99.225%		      Sum Accuracy:73.767% 

epoch 49
loss1=1.4612 loss2=2.2761 batch_id=749: 100%|██████████| 750/750 [00:14<00:00, 52.81it/s]
Test Set: Total Loss:3.760     MNIST Accuray:99.183%		      Sum Accuracy:73.533% 
```










