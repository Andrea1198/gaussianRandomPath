from src.functions import *
import numpy as np
import matplotlib.pyplot as plt
import imageio
steps       = 30
counts      = 2000
Nwalkers    = 10000
x0          = 0.
D           = 1.
norm        = 1./np.sqrt(2 * D * np.pi)
deltaTime   = 1e-2
time        = 0.
nbins       = 101

x           = np.zeros(Nwalkers)

# Bounds for periodic condition
a           =-5.
b           = 5.
L           = b - a
files       = []
q           = 0

# Stochastic Process
for i in range(steps):
# with open(files[i], 'w') as f:
    for j in range(counts):
        x   = calcTraj(x, D, Nwalkers, norm, deltaTime, time, L, a, b, x0)

    files.append("./files/" + str(i)+".png")
    plt.hist(x, nbins, density=True)
    plt.xlim(a, b)
    plt.ylim(-0.05, 1.)
    plt.savefig(files[i])
    plt.close()


with imageio.get_writer('particels.gif', mode='I') as writer:
    for filename in files:
        image = imageio.imread(filename)
        writer.append_data(image)
