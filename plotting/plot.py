import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

matplotlib.style.use("seaborn")

fileName = sys.argv[1]
dataset = sys.argv[2]
runPlots = int(sys.argv[3])

kValues = [32,64,128,256,512,1024]

print(f"Reading file: {fileName}")
namesArr = ["implementation","keyImplementation","sketchDim","k","THREAD_QUEUE_SIZE","bucketKeyBits","tables","WITH_TQ_OR_BUFFER","preprocessTime","constructionTime","scanTime","recall","avgDistance"]
data = np.genfromtxt(fileName, dtype=None, names=namesArr, delimiter=",")
data = data[data["scanTime"].argsort()]
# print(data)

yMajorTicks = np.arange(0,1.1,0.2)

def labelsLogTQ(x, pos):
    return "test"

def plotBufferVsSort(ax, dataSort, dataBuffer, k):
    xMajorTicks = np.array([2,4,8,16,32,64,128])
    ax.semilogx(dataSort["THREAD_QUEUE_SIZE"], dataSort["scanTime"], 'C1', label="Sort")
    ax.semilogx(dataBuffer["THREAD_QUEUE_SIZE"], dataBuffer["scanTime"], 'C2', label="Buffer")

    ax.set_xlabel("Thread queue size")
    ax.set_ylabel("Scan time (ms)")
    ax.set_title(f"Buffer versus Sort k={k}")
    ax.set_xticks(xMajorTicks)
    ax.set_xticklabels(xMajorTicks)
    ax.legend()


def plotRecallVsTime(ax, dataK, k):
    ax.plot(dataK[dataK["implementation"]==2]["scanTime"], dataK[dataK["implementation"]==2]["recall"],'C5', label="MEMOP")
    ax.plot(dataK[dataK["implementation"]==3]["scanTime"], dataK[dataK["implementation"]==3]["recall"],'C1', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["scanTime"], dataK[dataK["implementation"]==4]["recall"],'C2', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["scanTime"], dataK[dataK["implementation"]==5]["recall"],'C3', label="MINHASH1BIT")
    # ax.plot(dataK[dataK["implementation"]==6]["scanTime"], dataK[dataK["implementation"]==6]["recall"],'C4', label="JL")

    ax.set_ylabel("Recall")
    ax.set_xlabel("Scan time (ms)")
    ax.set_title(f"{dataset} results k={k}")
    ax.set_yticks(yMajorTicks)
    ax.legend()


def plotRecallVsBits(ax, dataK, k):
    ax.plot(dataK[dataK["implementation"]==3]["sketchDim"]*32, dataK[dataK["implementation"]==3]["recall"],'C1', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["sketchDim"]*8, dataK[dataK["implementation"]==4]["recall"],'C2', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["sketchDim"]*32, dataK[dataK["implementation"]==5]["recall"],'C3', label="MINHASH1BIT")
    # ax.plot(dataK[dataK["implementation"]==6]["sketchDim"], dataK[dataK["implementation"]==6]["recall"],'C4', label="JL")

    ax.set_ylabel("Recall")
    ax.set_xlabel("Number of bits")
    ax.set_title(f"{dataset} sketch results k={k}")
    ax.set_yticks(yMajorTicks)
    ax.legend()

def runPlotRecallVsTimeData(): 
    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotRecallVsTime(axes, data[data["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}recallvstime{currentK}k.png")

    
def runPlotRecallVsBits():
    dataSort = data[data["sketchDim"].argsort()]
    
    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotRecallVsBits(axes, dataSort[dataSort["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}recallvsbits{currentK}k.png")

def runPlotBufVsSort():
    dataSortTQ = data[data["THREAD_QUEUE_SIZE"].argsort()]
    dataMemOptimized = dataSortTQ[dataSortTQ["implementation"]==2]
    dataBuffer = dataMemOptimized[dataMemOptimized["WITH_TQ_OR_BUFFER"]==0]
    dataSort = dataMemOptimized[dataMemOptimized["WITH_TQ_OR_BUFFER"]==1]

    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotBufferVsSort(axes, dataSort[dataSort["k"]==currentK], dataBuffer[dataBuffer["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}bufvssort{currentK}k.png")


def runAll():
    runPlotBufVsSort()
    runPlotRecallVsTimeData()


# Main 

switcher = {
    0: runPlotBufVsSort,
    1: runPlotRecallVsTimeData,
    2: runPlotRecallVsBits
}

print(f"Running {runPlots}")
runner = switcher.get(runPlots, runAll)

runner()