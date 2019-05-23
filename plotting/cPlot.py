import sys
import matplotlib
# matplotlib.use("SVG")

import matplotlib.pyplot as plt
import numpy as np

MajorTicks = np.arange(0,1.01,0.1)

def optimalBound(c):
    val = 1 / (2 * (c**2) - 1)
    print(str(val))
    return val

def genData():
    
    c = 1.0
    arr = np.array([c, optimalBound(c)])
    c = c + 0.05
    while c < 3.0:
        bound = optimalBound(c)
        arr = np.append(arr, np.array([c, bound]), 0) 
        c = c + 0.05

    print(arr)
    return arr

def plot():
    c = 1.0
    y = np.array([c])
    while c < 3.0:
        c = c + 0.05
        y = np.append(y, np.array([c]), 0) 

    x = 1 / (2 * (y**2) - 1)
    plt.title('Relationship between c and rho')
    plt.ylabel('rho')
    plt.xlabel('c')
    plt.yticks(np.arange(min(y), max(y)+1, 0.2))
    plt.plot(x,y) 
    plt.savefig('./plots/rhoandc.png')
    

#MAIN
matplotlib.style.use("seaborn")
plot()
