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

X = 1
D = 2
dx = 0.005
nx = int(round(X/dx))
spatial = np.linspace(0, X, nx+1, dtype=np.float32)
x = np.linspace(0, X, nx+1)

T = 4
dt = 0.0001
nt = int(round(T/dt))
temporal = np.linspace(0, T, nt, dtype=np.float32)

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
    dy = 0.000005
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

        if i % 1000 == 0:
            Gamma[sn, :]=gamma[i,:]
            sn+=1
    Gamma[nx,:]=gamma[ny,:]
    return Gamma

############# DATASET GENERATION. Takes about 2 minutes.
kbudarr = []
gammaarr = []
karr = []
lamValArr = []
lamArr = np.random.uniform(5, 10, 2000)
print(lamArr.shape)
for i in range(2000):
    # init_cond = np.zeros(nx+1)
    lam = np.zeros(nx+1)
    for j in range(nx+1):
        # init_cond[j] = 10
        # Chebyshev polynomials
        lam[j] = 10.2+2*math.cos(lamArr[i]*math.acos(spatial[j]))

    k = solveIntegralk(X, nx, spatial, lam)
    kbud=k[-1,:]
    gamma = solveIntegralGamma(k,D, X, nx, x, lam)
    karr.append(k)
    kbudarr.append(kbud)
    gammaarr.append(gamma)
    lamValArr.append(lam)
    print("i=", i)

def solveIntegralGammay(k,Gamma, nx):
    gamma_y=(Gamma[:,nx]-Gamma[:,nx-1])/(dx)
    k_y=(k[nx][nx]-k[nx][nx-1])/(dx)
    gamma_y[0]=k_y
    return gamma_y  

x = np.array(lamValArr, dtype=np.float32)
y = np.array(karr, dtype=np.float32)
y1 = np.array(kbudarr, dtype=np.float32)
y2 = np.array(gammaarr, dtype=np.float32)

x = x.reshape(x.shape[0], -1)
y = y.reshape(y.shape[0], -1)
y1 = y1.reshape(y1.shape[0], -1)
y2 = y2.reshape(y2.shape[0], -1)

np.savetxt("lamValArr.dat", x)
np.savetxt("karr.dat", y)
np.savetxt("kbudarr.dat", y1)
np.savetxt("gammaarr.dat", y2)

gamma_yarr = []
for i in range(1000):
    gamma_y = solveIntegralGammay(karr[i,:].reshape(201, 201), gammaarr[i,:].reshape(201, 201), nx)
    gamma_yarr.append(gamma_y)
    print("i=", i)

