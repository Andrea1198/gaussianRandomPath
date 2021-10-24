from numpy import zeros, array, sqrt, log, cos, sin, pi, sum, abs, e
from numpy.random import random
from numba import jit
import matplotlib.pyplot as plt

# Generating random rumbers w/ gaussian distribution
@jit(nopython=True, fastmath=True)
def createRandomNums(N, deviation):
    y           = zeros(2*N)
    x1          = random(N) * deviation
    x2          = random(N)
    # Memorizing values used more times
    logx1       = -2. * log(x1) * deviation
    sqrlogx     = sqrt(logx1)
    x2pi        = x2*2.*pi
    y[0:N]      = sqrlogx * cos(x2pi)
    # It seems that sqrt is faster than sin
    y[N:2*N]    = sqrlogx * sin(x2pi)
    # y[N:2*N]    = sqrt(logx1-y[0:N]**2)
    return y

@jit(nopython=True, fastmath=True)
def createTraj(x0, y, steps):
    x   = zeros(steps+1)
    x[0]= x0
    for i in range(1, steps+1):
        x[i] = x[i-1] + y[i-1]
    return x

def plotTraj(x, N):
    for i in range(N):
        plt.plot(x[i,:])
    plt.show()

@jit(nopython=True, fastmath=True)
def smoluchovski(f, gamma, v0, steps):
    v   = zeros(steps+1)
    v[0]= v0
    for i in range(1, steps+1):
        v[i]    = v[i-1] - gamma*v[i-1] + f[i]
    return v

# Calculating statistical variables
@jit(nopython=True, fastmath=True)
def calculateVariables(y, N, res):
    somma       = sum(y)
    mean        = somma/N
    # Dividing in intervals
    ran         = [min(y), max(y)]
    interval    = ran[1]-ran[0]
    interval   /= res
    dy  = abs(y-mean)
    dy *= dy
    dev = sum(dy)
    dev/= (N-1)
    return mean, dev, interval

# Creating Histogram for visualization
# @jit(nopython=True)
def createHist(y, res):
    from numpy import max, min
    maxY    = max(y[:])
    minY    = min(y[:])
    dy      = (maxY - minY)/res
    N       = y.size
    hist        = zeros(res)
    for i in range(N):
        n   = int(y[i]/dy + res/2-1)
        if n < res and n >= 0:
            hist[n] += 1
    return hist
