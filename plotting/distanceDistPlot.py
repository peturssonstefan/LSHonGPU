import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
import numpy as np

def readData(fileName, numElems):
    print(f"Reading file: {fileName}")
    f = open(fileName)
    arr = np.array([1])
        
    counter = 0
    for line in f:
        if(counter > numElems): break
        if line.startswith('Q') != True and (np.random.rand() < .2):
            val = float(line)
            # if val >= 0 and val <= 1:
            arr = np.append(arr, np.array([float(line)]), 0)
        counter = counter + 1

    print(arr)
    return arr


def createGraph(data, fileName):
    fig , axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    axes.grid(axis='y', alpha=0.75)
    axes.hist(data, 80, color='#1e90ff', alpha=0.7, rwidth=0.85)
    axes.set_xlabel('Distance')
    axes.set_ylabel('Frequency')
    axes.set_title("Distance Distribution on Glove with Angular Distance")

    fileName = fileName.replace("txt","png")
    plt.savefig(f"{fileName}")
    plt.close(fig)



#Main 

matplotlib.style.use("seaborn")
fileName = sys.argv[1]
numElems = int(sys.argv[2])
data = readData(fileName, numElems)
createGraph(data, fileName)