import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use("seaborn")

fileName = sys.argv[1]
print(f"Reading file: {fileName}")

namesArr = ["implementation","keyImplementation","sketchDim","k","THREAD_QUEUE_SIZE","bucketKeyBits","tables","WITH_TQ_OR_BUFFER","preprocessTime","constructionTime","scanTime","recall","avgDistance"]
data = np.genfromtxt(fileName, dtype=None, names=namesArr, delimiter=",")
data = data[data["scanTime"].argsort()]
# print(data)

yMajorTicks = np.arange(0,1.1,0.2)


def plotData(ax, dataK, k):
    ax.plot(dataK[dataK["implementation"]==3]["scanTime"], dataK[dataK["implementation"]==3]["recall"],'C1', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["scanTime"], dataK[dataK["implementation"]==4]["recall"],'C2', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["scanTime"], dataK[dataK["implementation"]==5]["recall"],'C3', label="MINHASH1BIT")
    # ax.plot(dataK[dataK["implementation"]==6]["scanTime"], dataK[dataK["implementation"]==6]["recall"],'C4', label="JL")

    ax.set_ylabel("Recall")
    ax.set_xlabel("Scan time")
    ax.set_title(f"Glove results k={k}")
    ax.set_yticks(yMajorTicks)
    ax.legend()


fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

currentK = 32
for ax in axes.flatten():
    plotData(ax, data[data["k"]==currentK], currentK)
    currentK*=2


# plt.savefig("test.svg")
plt.show()