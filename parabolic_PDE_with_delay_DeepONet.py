

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
# import deepxde as dde
import time

def solvePDE(I, a, L, dt,F, T, lam, gamma, gamma_y):
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt)
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    U = np.zeros((2*(Nx+1), Nt))
    U[0:Nx+1,0]=I
    u = np.zeros((Nx+1,Nt))
    v = np.zeros((Nx+1,Nt))

    A_matrix = np.zeros((2*len(x), 2*len(x)))
    At_matrix = np.zeros((2*len(x), 2*len(x)))

    for j in range(1, len(x)-1):
        A_matrix[j, j-1] = dt/ (2 * dx**2)
        A_matrix[j, j] = (1 - dt/dx**2) +dt*lam[j]/2
        A_matrix[j, j+1] = dt/ (2 * dx**2)
        A_matrix[j+Nx, j+Nx] = 1 - dt/(2*D*dx)
        A_matrix[j+Nx, j+Nx+1] = dt/(2*D*dx)

        At_matrix[j, j-1] = -  dt/ (2 * dx**2)
        At_matrix[j, j] = (1 + dt/dx**2) - dt*lam[j]/2
        At_matrix[j, j+1] = - dt/ (2 * dx**2)
        At_matrix[j+Nx, j+Nx] = 1 + dt/(2*D*dx)
        At_matrix[j+Nx, j+Nx+1] =- dt/(2*D*dx)

    A_matrix[2*Nx, 2*Nx] = 1 - dt/(2*D*dx)
    A_matrix[2*Nx, 2*Nx+1] = dt/(2*D*dx)
    A_matrix[0, 0] = 1
    A_matrix[Nx, Nx+1] = 1/2
    At_matrix[2*Nx, 2*Nx] = 1 + dt/(2*D*dx)
    At_matrix[2*Nx, 2*Nx+1] =- dt/(2*D*dx)
    At_matrix[0, 0] = 1
    At_matrix[Nx, Nx] = 1
    At_matrix[Nx, Nx+1] = -1/2

    Al=np.zeros((1,Nx+1))
    Al[0, 0]=1/3
    Al[0, Nx]=1/3
    for i in range(1, Nx):
        if i % 2==0:
            Al[0, i]=2/3
        else:
            Al[0, i]=4/3
    Al=Al*dx
    np.savetxt("Al.dat", Al)
    E=np.ones((1,2*(Nx+1)))
    E[0,0:Nx+1]=gamma[Nx,:]*Al
    for i in range(0, nx+1):
        E[0,Nx+1+i]=-D*gamma_y[Nx-i]*Al[0,i]
    np.savetxt("E.dat", E)
    At_matrix[2*Nx+1,:]=-E/2
    At_matrix[2*Nx+1,2*Nx+1]=At_matrix[2*Nx+1,2*Nx+1]+1
    A_matrix[2*Nx+1,:]=E/2


    for i in range(1, Nt):
        # print("i=", i)
        if i % int(Nt/10) == 0:
            print("i", i, "/", nt)
        # Compute u at inner mesh points
        U[:,i] = np.matmul(np.linalg.inv(At_matrix), np.matmul(A_matrix, U[:,i-1]))
        # print("u:", u)

    u=U[0:Nx+1,:]
    v=U[Nx+2:2*Nx+1,:]
    return u,x

def solveIntegralk(X, nx, x, lam):
    k = np.zeros((len(x), len(x)))
    # First we calculate a at each timestep
    a = lam

    # FD LOOP
    k[1][1] = -(a[1] + a[0]) * dx / 4
    for i in range(1, len(x)-1):
        k[i+1][0] = 0
        k[i+1][i+1] = k[i][i]-dx/4.0*(a[i-1] + a[i])
        k[i+1][i] = k[i][i] - dx/2 * a[i]
        for j in range(1, i):
                k[i+1][j] = -k[i-1][j] + k[i][j+1] + k[i][j-1] + a[j]*(dx**2)*(k[i][j+1]+k[i][j-1])/2
    return k


def solveIntegralGamma(k,D, X, nx, x, lam):
    X = 1
    dy = 0.000001
    ny= int(round(X/dy))
    y = np.linspace(0, X, ny+1)
    print("ny=", ny)

    gamma = np.zeros((len(y), len(x)))
    Gamma = np.zeros((len(x), len(x)))
    sn=0
    gamma[0, :]=k[nx, :]

    for i in range(0, len(y)-1):
        for j in range(1, nx):
                gamma[i+1][j] =  D*(dy/(dx**2)+dy*lam[j]/2)*gamma[i][j+1] + (D*dy/(dx**2))* gamma[i][j-1] + (1-2*D*dy/(dx**2)+D*dy*lam[j]/2)*gamma[i][j]

        if i % 5000 == 0:
            Gamma[sn, :]=gamma[i,:]
            print("sn=", sn)
            sn+=1
    Gamma[nx,:]=gamma[ny,:]
    return Gamma

def solveIntegralGammay(k,Gamma, nx):
    gamma_y=(Gamma[:,nx]-Gamma[:,nx-1])/(dx)
    k_y=(k[nx][nx]-k[nx][nx-1])/(dx)
    gamma_y[0]=k_y
    return gamma_y


def openLoop(u, kernel, nx, dx):
    return 0


X = 1
D = 2
dx = 0.005
nx = int(round(X/dx))
spatial = np.linspace(0, X, nx+1, dtype=np.float32)
x = np.linspace(0, X, nx+1)

T = 10
dt = 0.0001
nt = int(round(T/dt))
temporal = np.linspace(0, T, nt, dtype=np.float32)



# Build out grid for k
# Parameters
epochs =500
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5
learning_rate = 0.001
step_size= 50
modes=12
width=32

grids = []
grids.append(spatial)
grids.append(spatial)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = torch.from_numpy(grid).cuda()
print("a=", grid.shape)


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
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16,32, 5, stride=2)
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



def zeroToNan(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if j >= i:
                x[i][j] = float('nan')
    return x

# PDE L2 Error
def getPDEl2(u, uhat):
    pdeError = np.zeros(nt-1)
    for i in range(1, nt):
        error = 0
        for j in range(nx):
            error += (u[i][j] - uhat[i][j])**2
        error = np.sqrt(error*0.01)
        pdeError[i-1] = error
    return pdeError


# Build out two examples
uarr = []
karr = []
khatarr = []
uopenarr = []
uhatarr = []
gammaarr=[]  
gammahatarr=[]
gamma_yhatarr=[]
gamma_yarr=[]
lambdaarr=[]

lamarr = [5,10]  #lamarr=5 is 1

model1=torch.load("parabolicKernelLambdaToK").cuda()
model2=torch.load("parabolicKernelGamma").cuda()
print(lamarr)
for i in range(1):
    init_cond = np.zeros(nx+1)
    lam = np.zeros(nx+1)
    for j in range(nx+1):
        init_cond[j] = 10
        lam[j] = 10.2+2*math.cos(lamarr[i]*math.acos(spatial[j]))
    
    init_cond[0] = 0
    k = solveIntegralk(X, nx, spatial, lam)
    

    xval = []
    for j in range(len(lam)):
        xval.append(lam)

    xval = np.array(xval, dtype=np.float32)
    print(xval.shape)
    xval = torch.from_numpy(xval.reshape(1, (nx+1)**2)).cuda()
    khat = model1((xval, grid))
    khat = khat.cpu().detach().numpy().reshape(nx+1, nx+1)
    
    kval = []
    for j in range(len(lam)):
        kval.append(khat[-1, :])

    kval = np.array(kval, dtype=np.float32)
    print(kval.shape)
    kval = torch.from_numpy(kval.reshape(1, (nx+1)**2)).cuda()
    Gval = torch.stack([xval,kval], axis=1)
    Gval= torch.reshape(Gval, (Gval.shape[0], -1, nx+1, nx+1))
    print(Gval.shape)
    gammahat = model2((Gval, grid))
    gammahat = gammahat.cpu().detach().numpy().reshape(nx+1, nx+1)
    # np.savetxt("gmmahat.dat", gammahat)

    Gamma = solveIntegralGamma(k,D, X, nx, x, lam)
    gamma_y = solveIntegralGammay(k, Gamma, nx)
    gamma_yhat = solveIntegralGammay(khat, gammahat, nx)

    uu, s2= solvePDE(init_cond, 1, 1, dt, dt/dx**2, T, lam, Gamma, gamma_y)
    u = uu.T
    print(u.shape)
    
    uu3, s3= solvePDE(init_cond, 1, 1, dt, dt/dx**2, T, lam, gammahat, gamma_yhat)
    u3 = uu3.T
    print(u3.shape)

    karr.append(k)
    khatarr.append(khat)
    uarr.append(u)
    uhatarr.append(u3)
    gammaarr.append(Gamma)
    gammahatarr.append(gammahat)
    gamma_yhatarr.append(gamma_yhat)
    gamma_yarr.append(gamma_y)
    lambdaarr.append(lam)  
    np.savetxt("uarr1.dat", u)
    np.savetxt("uhatarr1.dat", u3)
    np.savetxt("karr1.dat", k)  
    np.savetxt("khatarr1.dat", khat)
    np.savetxt("gammaarr1.dat", Gamma)
    np.savetxt("gammahatarr1.dat", gammahat)
    np.savetxt("gamma_yarr1.dat", gamma_y)
    np.savetxt("gamma_yhatarr1.dat", gamma_yhat)
    np.savetxt("lambdaarr1.dat", lam)