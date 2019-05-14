import sys
import matplotlib
from scipy.stats import binom
import matplotlib.pyplot as plt

import math

import numpy as np

matplotlib.style.use("seaborn")


def getBinomialDistribution(n, p0, k):
    p1 = 1-p0
    return binom.pmf(k, n, p1)

def angularProbability(r):
    radians = math.radians(r)
    return 1 - (radians/math.pi)

numOfBits = int(sys.argv[1])
numberOfOnes = np.arange(numOfBits+1)

r1 = float(sys.argv[2])
c = float(sys.argv[3])
r2 = r1 * c

pClose = angularProbability(r1)
pFar = angularProbability(r2)

print(f"pClose={pClose} pFar={pFar}")

plt.plot(numberOfOnes, getBinomialDistribution(numOfBits, pClose, numberOfOnes), "-", label="Close Points")
plt.plot(numberOfOnes, getBinomialDistribution(numOfBits, pFar, numberOfOnes), "-", label="Far Points")
plt.xlabel("# number of bits set to 1")
plt.ylabel("Probabilty of # of bits set to 1")
plt.title(f"Binomial graph r={r1} c={c}")
plt.legend()

plt.show()