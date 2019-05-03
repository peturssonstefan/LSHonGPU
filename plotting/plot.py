import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use("seaborn")

fileName = sys.argv[1]
dataset = sys.argv[2]
runPlots = int(sys.argv[3])
numQueries = int(sys.argv[4])

kValues = [32,64,128,256,512,1024]
tqSizes = [2,4,8,16,32,64,128]

print(f"Reading file: {fileName}")
namesArr = ["implementation","keyImplementation","sketchDim","k","THREAD_QUEUE_SIZE","bucketKeyBits","tables","WITH_TQ_OR_BUFFER","preprocessTime","constructionTime","scanTime","recall","avgDistance"]
data = np.genfromtxt(fileName, dtype=None, names=namesArr, delimiter=",")
data = data[data["scanTime"].argsort()]
# print(data)

# transform data scantime to queries per second
data["scanTime"] = numQueries/(data["scanTime"]/1000) 

yMajorTicks = np.arange(0,1.1,0.2)


def plotBufferVsSort(ax, dataSort, dataBuffer, k):
    xMajorTicks = np.array([2,4,8,16,32,64,128])
    ax.semilogx(dataSort["THREAD_QUEUE_SIZE"], dataSort["scanTime"], 'C1', label="Sort")
    ax.semilogx(dataBuffer["THREAD_QUEUE_SIZE"], dataBuffer["scanTime"], 'C2', label="Buffer")

    ax.set_xlabel("Thread queue size")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"Buffer versus Sort k={k}")
    ax.set_xticks(xMajorTicks)
    ax.set_xticklabels(xMajorTicks)
    ax.legend()


def plotRecallVsTime(ax, dataK, k, tqSize):
    ax.plot(dataK[dataK["implementation"]==2]["recall"],dataK[dataK["implementation"]==2]["scanTime"],'o', label="MEMOP")
    ax.plot(dataK[dataK["implementation"]==3]["recall"],dataK[dataK["implementation"]==3]["scanTime"],'o', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["recall"],dataK[dataK["implementation"]==4]["scanTime"],'o', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["recall"],dataK[dataK["implementation"]==5]["scanTime"],'o', label="MINHASH1BIT")
    ax.plot(dataK[dataK["implementation"]==6]["recall"],dataK[dataK["implementation"]==6]["scanTime"],'o', label="JL")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{dataset} results k={k} ThreadQueueSize={tqSize}")
    #ax.set_yticks(yMajorTicks)
    ax.legend()


def plotRecallVsBits(ax, dataK, k, tqSize):
    ax.plot(dataK[dataK["implementation"]==3]["sketchDim"]*32, dataK[dataK["implementation"]==3]["recall"],'o', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["sketchDim"]*8, dataK[dataK["implementation"]==4]["recall"],'o', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["sketchDim"]*32, dataK[dataK["implementation"]==5]["recall"],'o', label="MINHASH1BIT")
    ax.plot(dataK[dataK["implementation"]==6]["sketchDim"], dataK[dataK["implementation"]==6]["recall"],'o', label="JL")

    ax.set_ylabel("Recall")
    ax.set_xlabel("Number of bits")
    ax.set_title(f"{dataset} sketch results k={k} ThreadQueueSize={tqSize}")
    ax.set_yticks(yMajorTicks)
    ax.legend()


def plotDistanceRatioVsScanTime(ax, dataK, k, tqSize):
    ax.plot(dataK[dataK["implementation"]==2]["avgDistance"],dataK[dataK["implementation"]==2]["scanTime"],'o', label="MEMOP")
    ax.plot(dataK[dataK["implementation"]==3]["avgDistance"],dataK[dataK["implementation"]==3]["scanTime"],'o', label="SIMHASH")
    ax.plot(dataK[dataK["implementation"]==4]["avgDistance"],dataK[dataK["implementation"]==4]["scanTime"],'o', label="MINHASH")
    ax.plot(dataK[dataK["implementation"]==5]["avgDistance"],dataK[dataK["implementation"]==5]["scanTime"],'o', label="MINHASH1BIT")
    ax.plot(dataK[dataK["implementation"]==6]["avgDistance"],dataK[dataK["implementation"]==6]["scanTime"],'o', label="JL")
    
    ax.set_xlabel("Distance ratio")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{dataset} results k={k} ThreadQueueSize={tqSize}")
    #ax.set_yticks(yMajorTicks)
    ax.legend()


def runPlotRecallVsTimeData():     
    for tqSize in tqSizes:
        dataTQ = data[data["THREAD_QUEUE_SIZE"]==tqSize]
        for currentK in kValues:
            fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
            plotRecallVsTime(axes, dataTQ[dataTQ["k"]==currentK], currentK, tqSize)
            plt.savefig(f"plots/{dataset}recallvstime{currentK}k_{tqSize}tq.png")
            plt.close(fig)


def runPlotDistanceRatioVsTime():
    for tqSize in tqSizes:
        dataTQ = data[data["THREAD_QUEUE_SIZE"]==tqSize]
        for currentK in kValues:
            fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
            plotDistanceRatioVsScanTime(axes, dataTQ[dataTQ["k"]==currentK], currentK, tqSize)
            plt.savefig(f"plots/{dataset}distancevstime{currentK}k_{tqSize}tq.png")
            plt.close(fig)

def runPlotRecallVsBits():
    dataSort = data[data["sketchDim"].argsort()]
    
    for tqSize in tqSizes:
        dataTQ = dataSort[dataSort["THREAD_QUEUE_SIZE"]==tqSize]
        for currentK in kValues:
            fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
            plotRecallVsBits(axes, dataTQ[dataTQ["k"]==currentK], currentK, tqSize)
            plt.savefig(f"plots/{dataset}recallvsbits{currentK}k_{tqSize}tq.png")
            plt.close(fig)

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
    runPlotRecallVsBits()
    runPlotDistanceRatioVsTime()


# Main 

switcher = {
    0: runPlotBufVsSort,
    1: runPlotRecallVsTimeData,
    2: runPlotRecallVsBits,
    3: runPlotDistanceRatioVsTime
}

print(f"Running {runPlots}")
runner = switcher.get(runPlots, runAll)

runner()