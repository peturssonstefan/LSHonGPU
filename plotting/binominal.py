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

xMajorTicks = np.arange(0,numOfBits+1, numOfBits/10)

plt.plot(numberOfOnes, getBinomialDistribution(numOfBits, pClose, numberOfOnes), "-", label="Close Points")
plt.plot(numberOfOnes, getBinomialDistribution(numOfBits, pFar, numberOfOnes), "-", label="Far Points")
plt.xlabel("Number of bits that differ")
plt.ylabel("Probabilty for given amount of bits to differ")
plt.title(f"Binomial distribution for Simhash")

plt.xticks(xMajorTicks)

plt.legend()


plt.savefig(f"binomial_bits_{numOfBits}_r1_{r1}c_{c}.png")