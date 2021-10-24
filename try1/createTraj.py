from numpy import zeros, array, sum, abs
# To generate random gaussian numbers
from numpy.random import normal
from math import sqrt
from time import process_time
from src.functions import *
import matplotlib.pyplot as plt

# import multiprocessing as mp

N           = int(1e4)
res         = 101
S           = 0
k           = 1.
T           = 1.
m           = 1.
gamma       = 1.
deltaT      = 1.
dt          = 1e-2
steps       = int(deltaT//dt)
y           = zeros((N, steps))
x0          = 20.
# D           = k * T / m / gamma
D           = 1.
sqrtD       = sqrt(D)

print("# Creating random numbers..")
tic         = process_time()
# Random Numbers w/ Gaussian distribution
y           = normal(0, sqrt(D), (N, steps)) / sqrt(2 * pi) / sqrtD
print("# Time:                  ", process_time()-tic)

x   = zeros((N, steps+1))
print("# Creating trajectory..")
tic                     = process_time()
for i in range(N):
    x[i,:]              = createTraj(x0, y[i,:], steps)
print("# Time:                  ", process_time()-tic)

# plotTraj(x, N)

# Show the final distribution of positions
plt.hist(x[:,-1], res, density=True)
plt.show()



# Attempting parallization
# NCPU = mp.cpu_count()
# pool    = mp.Pool(NCPU)
# y                       = array([pool.apply(createRandomNums, args=(int(N/2/NCPU))) for i in range(NCPU)])
# mean, dev, interval     = array([pool.apply(calculateVariables, args=(y, N/NCPU, res)) for i in range(NCPU)])
# hist                    = array([pool.apply(createHist, args=(N/NCPU, y, interval, res)) for i in range(NCPU)])
# pool.close()
