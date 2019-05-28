import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use("seaborn")

sketchFileName = sys.argv[1]
lshFileName = sys.argv[2]
dataset = sys.argv[3]
runPlots = int(sys.argv[4])
numQueries = int(sys.argv[5])

kValues = [32,128,1024]
tqSizes = [2,4,8,16,32,64,128]

def readData(fileName):
    print(f"Reading file: {fileName}")
    namesArr = ["implementation","keyImplementation","sketchDim","k","THREAD_QUEUE_SIZE","bucketKeyBits","tables","WITH_TQ_OR_BUFFER","preprocessTime","constructionTime","scanTime","recall","avgDistance"]
    data = np.genfromtxt(fileName, dtype=None, names=namesArr, delimiter=",")
    data = data[data["scanTime"].argsort()]
    data = data[data["recall"].argsort()]
    # print(data)

    # transform data scantime to queries per second
    data["scanTime"] = numQueries/(data["scanTime"]/1000)     
    return data

MajorTicks = np.arange(0,1.01,0.1)


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


def plotRecallVsTime(ax, dataK, k):
    tqSizesLocal = [8,32,128]

    if k == 1024:
        tqSizesLocal = [64,128]

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.loglog(dataTq[dataTq["implementation"]==2]["recall"],dataTq[dataTq["implementation"]==2]["scanTime"],':', label=f"MEMOP {tqSize}")
    
    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.loglog(dataTq[dataTq["implementation"]==3]["recall"],dataTq[dataTq["implementation"]==3]["scanTime"],'o', label=f"SIMHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.loglog(dataTq[dataTq["implementation"]==4]["recall"],dataTq[dataTq["implementation"]==4]["scanTime"],'D', label=f"MINHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.loglog(dataTq[dataTq["implementation"]==5]["recall"],dataTq[dataTq["implementation"]==5]["scanTime"],"^", label=f"MINHASH1BIT {tqSize}", linestyle="-")
        
    ax.set_xlabel("Recall")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{dataset} results k={k}")
    #ax.set_yticks(MajorTicks)
    #ax.set_xticks(MajorTicks)
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plotRecallVsBits(ax, dataK, k):
    tqSizesLocal = [8,32,128]

    if k == 1024:
        tqSizesLocal = [64,128]

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.plot(dataTq[dataTq["implementation"]==3]["sketchDim"]*32,dataTq[dataTq["implementation"]==3]["recall"],'o', label=f"SIMHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.plot(dataTq[dataTq["implementation"]==4]["sketchDim"]*8,dataTq[dataTq["implementation"]==4]["recall"],'D', label=f"MINHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.plot(dataTq[dataTq["implementation"]==5]["sketchDim"]*32,dataTq[dataTq["implementation"]==5]["recall"],"^", label=f"MINHASH1BIT {tqSize}", linestyle="-")
        
    ax.set_ylabel("Recall")
    ax.set_xlabel("Number of bits")
    ax.set_title(f"{dataset} sketch results k={k}")
    ax.set_yticks(MajorTicks)
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plotDistanceRatioVsScanTime(ax, dataK, k):
    
    tqSizesLocal = [8,32,128]

    if k == 1024:
        tqSizesLocal = [64,128]

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.semilogy(dataTq[dataTq["implementation"]==2]["avgDistance"],dataTq[dataTq["implementation"]==2]["scanTime"],'X', label=f"MEMOP {tqSize}")
    
    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.semilogy(dataTq[dataTq["implementation"]==3]["avgDistance"],dataTq[dataTq["implementation"]==3]["scanTime"],'o', label=f"SIMHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.semilogy(dataTq[dataTq["implementation"]==4]["avgDistance"],dataTq[dataTq["implementation"]==4]["scanTime"],'D', label=f"MINHASH {tqSize}", linestyle="-")

    for tqSize in tqSizesLocal:
        dataTq = dataK[dataK["THREAD_QUEUE_SIZE"]==tqSize]
        ax.semilogy(dataTq[dataTq["implementation"]==5]["avgDistance"],dataTq[dataTq["implementation"]==5]["scanTime"],"^", label=f"MINHASH1BIT {tqSize}", linestyle="-")
    
    #ax.plot(dataK[dataK["implementation"]==6]["avgDistance"],dataK[dataK["implementation"]==6]["scanTime"],'o', label="JL")
    
    ax.set_xlabel("Distance ratio")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{dataset} results k={k}")

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def buildLshSeries(ax, data, implementation, bucketImplementation, tables, x, y, label, markerType):
    mask = (data["implementation"]==implementation) & (data["keyImplementation"]==bucketImplementation) & (data["tables"]==tables)
    ax.loglog(data[mask][x],data[mask][y],markerType, markersize=6, label=label)

def plotLshAlgorithmsRecallVsScanTime(ax, dataK, dataMemK, k, tqSize):
    #buildLshSeries(ax, dataK, 2, 3, "recall", "scanTime", "SIMHASH - scan", "o")
    buildLshSeries(ax, dataK, 3, 3, 2, "recall", "scanTime", "SIMHASH 2 Tables", "o-")
    buildLshSeries(ax, dataK, 3, 3, 4, "recall", "scanTime", "SIMHASH 4 Tables", "o-")
    #buildLshSeries(ax, dataK, 3, 3, 6, "recall", "scanTime", "SIMHASH 6 Tables", "o-")
    buildLshSeries(ax, dataK, 3, 3, 8, "recall", "scanTime", "SIMHASH 8 Tables", "o-")


    #buildLshSeries(ax, dataK, 2, 5, "recall", "scanTime", "1BITMINHASH - scan", "^")
    #buildLshSeries(ax, dataK, 5, 5, "recall", "scanTime", "1BITMINHASH - 1BitMINHASH", "^")    

    #buildLshSeries(ax, dataK, 2, 7, "recall", "scanTime", "CROSSPOLY - scan", "s")
    buildLshSeries(ax, dataK, 3, 7, 2, "recall", "scanTime", "CROSSPOLY 2 Tables", "s-")
    buildLshSeries(ax, dataK, 3, 7, 4, "recall", "scanTime", "CROSSPOLY 4 Tables", "s-")
   # buildLshSeries(ax, dataK, 3, 7, 6, "recall", "scanTime", "CROSSPOLY 6 Tables", "s-")
    buildLshSeries(ax, dataK, 3, 7, 8, "recall", "scanTime", "CROSSPOLY 8 Tables", "s-")

    #buildLshSeries(ax, dataK, 2, 4, "recall", "scanTime", "MINHASH - scan", "D")
    #buildLshSeries(ax, dataK, 5, 4, "recall", "scanTime", "MINHASH - 1BitMINHASH", "D")

    ax.plot(dataMemK[dataMemK["implementation"]==2]["recall"], dataMemK[dataMemK["implementation"]==2]["scanTime"], ":", label="Baseline Scan")
    ax.plot(dataMemK[dataMemK["implementation"]==3]["recall"], dataMemK[dataMemK["implementation"]==3]["scanTime"], "o:", markersize=6, label="Simhash Scan")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{dataset} LSH results k={k} ThreadQueueSize={tqSize}")
    #ax.set_yticks(MajorTicks)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #ax.set_xticks([])
    ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.NullFormatter())
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend()

def plotLshbucketBitsVsRecall(ax, dataK, dataMemK, k, tqSize, numTables):
    buildLshSeries(ax, dataK, 2, 3, "bucketKeyBits","recall", "SIMHASH - scan", "o")
    buildLshSeries(ax, dataK, 3, 3, "bucketKeyBits","recall", "SIMHASH - SIMHASH", "o")
    buildLshSeries(ax, dataK, 5, 3, "bucketKeyBits","recall", "SIMHASH - 1BITMINHASH", "o")

    buildLshSeries(ax, dataK, 2, 5, "bucketKeyBits","recall", "1BITMINHASH - scan", "^")
    buildLshSeries(ax, dataK, 3, 5, "bucketKeyBits","recall", "1BITMINHASH - SIMHASH", "^")
    buildLshSeries(ax, dataK, 5, 5, "bucketKeyBits","recall", "1BITMINHASH - 1BitMINHASH", "^")    

    buildLshSeries(ax, dataK, 2, 7, "bucketKeyBits","recall", "CROSSPOLY - scan", "s")
    buildLshSeries(ax, dataK, 3, 7, "bucketKeyBits","recall", "CROSSPOLY - SIMHASH", "s")
    buildLshSeries(ax, dataK, 5, 7, "bucketKeyBits","recall", "CROSSPOLY - 1BitMINHASH", "s")

    buildLshSeries(ax, dataK, 2, 4, "bucketKeyBits","recall", "MINHASH - scan", "D")
    buildLshSeries(ax, dataK, 3, 4, "bucketKeyBits","recall", "MINHASH - SIMHASH", "D")
    buildLshSeries(ax, dataK, 5, 4, "bucketKeyBits","recall", "MINHASH - 1BitMINHASH", "D")

    ax.set_xlabel("Bucket Key Bits")
    ax.set_ylabel("Recall")
    ax.set_title(f"{dataset} LSH results k={k} ThreadQueueSize={tqSize} Tables={numTables}")
    #ax.set_yticks(MajorTicks)
    ax.legend()

def runPlotLshRecallVsTime(data, lshData):
    for tqSize in tqSizes:
        lshMask = (lshData["THREAD_QUEUE_SIZE"]==tqSize) 
        lshDataTQ = lshData[lshMask]
        dataTq = data[data["THREAD_QUEUE_SIZE"]==tqSize]
        for currentK in kValues:
            fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
            plotLshAlgorithmsRecallVsScanTime(axes, lshDataTQ[lshDataTQ["k"]==currentK], dataTq[dataTq["k"]==currentK] ,currentK, tqSize)
            plt.savefig(f"plots/{dataset}lshrecallvstime{currentK}k_{tqSize}tq_{512}sketchDim.png")
            plt.close(fig)
        

def runPlotLshBucketBitsVsRecall(data, lshData):
    for numTables in [2, 4, 6, 8, 10]:
        for tqSize in tqSizes:
            lshMask = (lshData["THREAD_QUEUE_SIZE"]==tqSize) & (lshData["tables"]==numTables) & ((lshData["sketchDim"]==512) | (lshData["implementation"]==2)) 
            lshDataTQ = lshData[lshMask]
            dataTq = data[data["THREAD_QUEUE_SIZE"]==tqSize]
            for currentK in kValues:
                fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
                plotLshbucketBitsVsRecall(axes, lshDataTQ[lshDataTQ["k"]==currentK], dataTq[dataTq["k"]==currentK] ,currentK, tqSize, numTables)
                plt.savefig(f"plots/{dataset}lshbucketbitsvsrecall{currentK}k_{tqSize}tq_{numTables}tables.png")
                plt.close(fig)

def runPlotRecallVsTimeData(data, lshData):     
    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotRecallVsTime(axes, data[data["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}recallvstime{currentK}k.png")
        plt.close(fig)


def runPlotDistanceRatioVsTime(data, lshData):
    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotDistanceRatioVsScanTime(axes, data[data["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}distancevstime{currentK}k.png")
        plt.close(fig)

def runPlotRecallVsBits(data, lshData):
    dataSort = data[data["sketchDim"].argsort()]
    
    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotRecallVsBits(axes, dataSort[dataSort["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}recallvsbits{currentK}k.png")
        plt.close(fig)
        

def runPlotBufVsSort(data, lshData):
    dataSortTQ = data[data["THREAD_QUEUE_SIZE"].argsort()]
    dataMemOptimized = dataSortTQ[dataSortTQ["implementation"]==2]
    dataBuffer = dataMemOptimized[dataMemOptimized["WITH_TQ_OR_BUFFER"]==0]
    dataSort = dataMemOptimized[dataMemOptimized["WITH_TQ_OR_BUFFER"]==1]

    for currentK in kValues:
        fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        plotBufferVsSort(axes, dataSort[dataSort["k"]==currentK], dataBuffer[dataBuffer["k"]==currentK], currentK)
        plt.savefig(f"plots/{dataset}bufvssort{currentK}k.png")
        plt.close(fig)


def runAll(data, lshData):
    runPlotBufVsSort(data, lshData)
    runPlotRecallVsTimeData(data, lshData)
    runPlotRecallVsBits(data, lshData)
    runPlotDistanceRatioVsTime(data, lshData)
    runPlotLshRecallVsTime(data, lshData)
    runPlotLshBucketBitsVsRecall(data, lshData)


def lshDataClean(lshData):
    mask = (lshData["scanTime"] < 20000)
    lshData = lshData[mask]
    return lshData


def transformBucketKeyBitsColumn(data):
    mask = (data["keyImplementation"]==4)
    newData = data[mask]
    newData["bucketKeyBits"] = newData["bucketKeyBits"]*8

    for bimp in [3,5,7]:
        mask = (data["keyImplementation"]==bimp)
        toAppend = data[mask]
        toAppend["bucketKeyBits"] = toAppend["bucketKeyBits"]*16
        newData = np.append(newData, toAppend, 0)

    #newData.astype(data.dtype)

    return newData
    
def transformSketchDimColumn(data):
    mask = (data["implementation"]==3)
    newData = data[mask]
    newData["sketchDim"] = newData["sketchDim"]*32

    mask = (data["implementation"]==5)
    toAppend = data[mask]
    toAppend["sketchDim"] = toAppend["sketchDim"]*32

    newData = np.append(newData, toAppend, 0)

    mask = (data["implementation"]==2)
    toAppend = data[mask]
    newData = np.append(newData, toAppend, 0)
    

    #newData.astype(data.dtype)

    return newData 


# Main 

sketchedData = readData(sketchFileName)
lshDataFile = readData(lshFileName)


lshDataFile = transformBucketKeyBitsColumn(lshDataFile)
lshDataFile = transformSketchDimColumn(lshDataFile)

print(lshDataFile)

#lshDataFile = lshDataClean(lshDataFile)


switcher = {
    0: runPlotBufVsSort,
    1: runPlotRecallVsTimeData,
    2: runPlotRecallVsBits,
    3: runPlotDistanceRatioVsTime,
    4: runPlotLshRecallVsTime,
    5: runPlotLshBucketBitsVsRecall
}

print(f"Running {runPlots}")
runner = switcher.get(runPlots, runAll)

runner(sketchedData, lshDataFile)