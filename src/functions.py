from numba import jit
import numpy as np

def createHist(x, nbins):
    import numpy as np
    minX    = min(x)
    maxX    = max(x)
    L       = maxX - minX
    dx      = L / nbins
    hist    = np.zeros(nbins)
    for bin in range(nbins):
        mask1   = x > minX + bin * dx
        mask2   = x < minX + (bin + 1) * dx
        mask    = [mask1[i] and mask2[i] for i in range(len(x))]
        hist[bin] = len(x[mask])
    return hist


@jit(nopython=True, fastmath=True)
def f(x, x0, gamma):
    return (x0-x)/gamma


@jit(nopython=True, fastmath=True)
def calcTraj(x, D, Nwalkers, norm, deltaTime, time, L, a, b, x0):
    gamma   = 1.e7
    x      += np.random.normal(0., D, Nwalkers) * norm * deltaTime + f(x, x0, gamma)
    # Periodic Boundry Conditions
    mask    = x < a
    x[mask]+= L
    mask    = x > b
    x[mask]-= L
    time   += deltaTime
    return x
