import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# from utilities3 import LpLoss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
from functools import reduce
from functools import partial
import operator
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter
import deepxde as dde
import time


# Build out grid
# Parameters
epochs = 500
ntrain = 900
ntest = 100
batch_size = 32
gamma = 0.5
learning_rate = 0.001 # 0.001
step_size = 50
modes = 12
width = 32

X = 1
dx = 0.005
nx = int(round(X/dx))
spatial = np.linspace(0, X, nx+1, dtype=np.float32)

grids = []
grids.append(spatial)
grids.append(spatial)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = torch.from_numpy(grid).cuda()
print(grid.shape)

# Create train/test splits
x = np.loadtxt("x.dat", dtype=np.float32)
y1 = np.loadtxt("y1.dat", dtype=np.float32)
y = np.loadtxt("y2.dat", dtype=np.float32)
x = np.stack((x,y1), axis=1)
print(x.shape, y1.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
print("Data load success!")

def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv2d(2, 16, 5, stride=2) # 双输入
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = torch.nn.Linear(73728, 1028)
        self.fc2 = torch.nn.Linear(1028, 256)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.shape, self.shape))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define a sequential torch network for batch and trunk. Can use COV2D which we will show later
dim_x = 2
m = (nx+1)**2

branch = BranchNet(nx+1)

model = dde.nn.DeepONetCartesianProd([m, branch], [dim_x, 128, 256, 256], "relu", "Glorot normal").cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss = torch.nn.MSELoss()
train_lossArr = []
test_lossArr = []
time_Arr = []

print("epoch \ttime \t\t\ttrain_loss \t\t\ttest_loss \tlr")
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()
        x = torch.reshape(x, (x.shape[0], 2, nx+1, nx+1))

        optimizer.zero_grad()
        out = model((x, grid))

        lp = loss(out, y)
        lp.backward()

        optimizer.step()
        train_loss += lp.item()

    scheduler.step()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in testData:
            x, y = x.cuda(), y.cuda()

            out = model((x, grid))
            test_loss += loss(out, y).item()

    train_loss /= len(trainData)
    test_loss /= len(testData)

    train_lossArr.append(train_loss)
    test_lossArr.append(test_loss)

    t2 = default_timer()
    time_Arr.append(t2-t1)
    if ep%50 == 0:
        print("{} \t{} \t{} \t{} \t{}".format(ep, t2-t1, train_loss, test_loss, 
                                              optimizer.state_dict()['param_groups'][0]['lr']))

# Display Model Details
plt.figure()
plt.plot(train_lossArr, label="Train Loss")
plt.plot(test_lossArr, label="Test Loss")
plt.yscale("log")
plt.legend()
plt.savefig('Train_Test_Loss.png')
plt.show()

testLoss = 0
trainLoss = 0
with torch.no_grad():
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        trainLoss += loss(out, y).item()
        
    for x, y in testData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        testLoss += loss(out, y).item()
    
    
# cpu inference
time_Arr_cpu = []
for x, y in testData:
    x, y = x.cpu(), y.cpu()
    model_cpu = model.to("cpu")
    grid_cpu = grid.cpu()

    t1_cpu = default_timer()
    out = model_cpu((x, grid_cpu))
    t2_cpu = default_timer()
    time_Arr_cpu.append(t2_cpu-t1_cpu)


time_Arr_gpu = []
for x, y in testData:
    x, y = x.cuda(), y.cuda()
    model = model.to("cuda:0")

    t1_gpu = default_timer()
    out = model((x, grid))
    t2_gpu = default_timer()
    time_Arr_gpu.append(t2_gpu-t1_gpu)


print("Training Time:", sum(time_Arr))    
print("Avg Epoch Time:", sum(time_Arr)/len(time_Arr))
print("CPU Avg Caculate Time:", sum(time_Arr_cpu)/len(time_Arr_cpu))
print("GPU Avg Caculate Time:", sum(time_Arr_gpu)/len(time_Arr_gpu))
print("Final Testing Loss:", testLoss / len(testData))
print("Final Training Loss:", trainLoss / len(trainData))

torch.save(model, "parabolicKernel")