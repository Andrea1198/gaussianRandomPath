from numpy import zeros, array, sum, abs
from time import process_time
from src.functions import *
import matplotlib.pyplot as plt
# import multiprocessing as mp

N           = 10000
res         = 100
S           = 0
K           = 1.e-6
T           = 1e-1
m           = 1e-6
gamma       = 1.e1
deltaTime   = 2
dt          = 1e-3
v0          = 2.
x0          = 0.
A           = gamma * K * T / m
deviation   = 2 * A
steps       = int(deltaTime//dt)
f           = zeros((N, steps))
mean        = zeros(N)
dev         = zeros(N)
interval    = zeros(N)

print("# Creating random numbers..")
tic         = process_time()
for i in range(steps):
    f[:,i]  = createRandomNums(int(N/2), deviation)
print("# Time:                  ", process_time()-tic)

print("# Computating variables..")
tic         = process_time()
for i in range(N):
    mean[i], dev[i], interval[i]    = calculateVariables(f[i, :], N, res)

meanDev, devDev, intervalDev        = calculateVariables(dev, N, res)
print("2 * A                    : ", A)
print("Mean Standard Deviation  : ", meanDev)

print("# Time:                  ", process_time()-tic)

v   = zeros((N, steps+1))
print("# Creating velocities..")
tic                     = process_time()
for i in range(N):
    v[i,:]              = smoluchovski(f[i,:], gamma, v0, steps)
print("# Time:                  ", process_time()-tic)

x   = zeros((N, steps+1))
print("# Creating trajectory..")
tic                     = process_time()
for i in range(N):
    x[i,:]              = createTraj(x0, v[i,:], steps)
print("# Time:                  ", process_time()-tic)

plotTraj(x, N)
