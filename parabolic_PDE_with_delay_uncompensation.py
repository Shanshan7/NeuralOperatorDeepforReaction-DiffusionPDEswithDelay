
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
import time


X = 1
dx = 0.005
nx = int(round(X/dx))
spatial = np.linspace(0, X, nx+1, dtype=np.float32)
x = np.linspace(0, X, nx+1)

T =4
dt = 0.00001
nt = int(round(T/dt))
temporal = np.linspace(0, T, nt, dtype=np.float32)

def solvePDE(I, a, L, dt,F, T, lam, solveControl, kernel):
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt)
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros((Nt, Nx+1))

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u[0][i] = I[i]

    for i in range(1, int(round(2/dt))+1):
        if i % int(Nt/10) == 0:
            print("i", i, "/", nt)
        # Compute u at inner mesh points
        u[i][1:Nx] = u[i-1][1:Nx] +  \
                      F*(u[i-1][0:Nx-1] - 2*u[i-1][1:Nx] + u[i-1][2:Nx+1]) + dt*lam[1:Nx]*u[i-1][1:Nx]

        # Insert boundary conditions
        u[i][0] = 0;  u[i][Nx] = 0 # solveControl(u[i], kernel, Nx-1, dx)


    for i in range(int(round(2/dt))+1, Nt):
        if i % int(Nt/10) == 0:
            print("i", i, "/", nt)
        # Compute u at inner mesh points
        u[i][1:Nx] = u[i-1][1:Nx] +  \
                      F*(u[i-1][0:Nx-1] - 2*u[i-1][1:Nx] + u[i-1][2:Nx+1]) + dt*lam[1:Nx]*u[i-1][1:Nx]

        # Insert boundary conditions
        u[i][0] = 0;  u[i][Nx] =  solveControl(u[i-int(round(2/dt))][:], kernel, Nx-1, dx)
    return u, x

def solveIntegralFD(X, nx, x, lam):
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


def solveControl(u, kernel, nx, dx):
    return sum(kernel[-1][0:nx+1]*u[0:nx+1])*dx


def openLoop(u, kernel, nx, dx):
    return 0


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
lamarr = [5, 10]
print(lamarr)
for i in range(1):
    init_cond = np.zeros(nx+1)
    lam = np.zeros(nx+1)
    for j in range(nx+1):
        init_cond[j] = 10
        lam[j] =10.2+2*math.cos(lamarr[i]*math.acos(spatial[j]))#10*math.acos(spatial[j])
    k = solveIntegralFD(X, nx, spatial, lam)

    # Prepare Tensor solution
    u1, s2= solvePDE(init_cond, 1, 1, dt, dt/dx**2, T, lam, solveControl, k)
    np.savetxt("u_uncomp1_0816.dat", u1)

lamarr = [10, 5]
print(lamarr)
for i in range(1):
    init_cond = np.zeros(nx+1)
    lam = np.zeros(nx+1)
    for j in range(nx+1):
        init_cond[j] = 10
        lam[j] =10.2+2*math.cos(lamarr[i]*math.acos(spatial[j]))#10*math.acos(spatial[j])
    k = solveIntegralFD(X, nx, spatial, lam)

    # Prepare Tensor solution
    u, s2= solvePDE(init_cond, 1, 1, dt, dt/dx**2, T, lam, solveControl, k)
    np.savetxt("u_uncomp2_0816.dat", u)
    
res = 10
fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 2, wspace=0)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0)

subfig1 = fig.add_subplot(gs[0, 0], projection='3d')
subfig2 = fig.add_subplot(gs[0, 1], projection='3d')

subfig1.set_box_aspect((1, 1, 0.5))
subfig2.set_box_aspect((1, 1, 0.5))

subfig1.set_title(r"Closed-loop $u_{NO}(x, t)$ for $\gamma$ = 5")
subfig2.set_title(r"Closed-loop $u_{NO}(x, t)$ for $\gamma$ = 8")

meshx, mesht = np.meshgrid(spatial, temporal)

subfig1.plot_surface(mesht, meshx, u1, edgecolor="white", lw=0.2, rstride=res * 750, cstride=res,
                     alpha=1, color="black", shade=False, rasterized=True)
test = np.ones(int(nt))
vals = (u1.transpose())[-1]
subfig1.plot(temporal[1:], test[1:], vals[1:], color="red", lw=1.5, antialiased=True)
subfig1.view_init(20, 30)
subfig1.set_xlabel("Time")
subfig1.set_ylabel("x")
subfig1.set_zlabel(r"$u_{NO}(x, t)$", labelpad=3)
subfig1.zaxis.set_rotate_label(False)
subfig1.invert_xaxis()
subfig1.tick_params(axis='x', labelsize=6)
subfig1.tick_params(axis='y', labelsize=6)
subfig1.tick_params(axis='z', labelsize=6)
subfig1.zaxis.set_tick_params(labelsize=6)

subfig2.plot_surface(mesht, meshx, u, edgecolor="white", lw=0.2, rstride=res * 750, cstride=res,
                     alpha=1, color="black", shade=False, rasterized=True)
test = np.ones(int(nt))
vals = (u.transpose())[-1]
subfig2.plot(temporal[1:], test[1:], vals[1:], color="red", lw=1.5, antialiased=True)
subfig2.view_init(20, 30)
subfig2.set_xlabel("Time")
subfig2.set_ylabel("x")
subfig2.set_zlabel(r"$u_{NO}(x, t)$", labelpad=3)
subfig2.zaxis.set_rotate_label(False)
subfig2.invert_xaxis()
subfig2.tick_params(axis='x', labelsize=6)
subfig2.tick_params(axis='y', labelsize=6)
subfig2.tick_params(axis='z', labelsize=6)
subfig2.zaxis.set_tick_params(labelsize=6)

plt.show()
