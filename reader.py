
import sys

def main():
    goldenStandards = sys.argv[1]
    results = sys.argv[2]
    print("Comparing file " + results + " to gold standards in " + goldenStandards)
    queryToResReal = dict()
    queryToResTest = dict()
    f = open(goldenStandards)
    nReal = f.readline()
    kReal = f.readline() 
    k = int(kReal)
    n = int(nReal)
    meanReal = 0
    currentKey = 0
    #Read real data
    for x in f:
        res = x.split()
        if(res[0].startswith('q')):
            currentKey = int(res[1])
            queryToResReal[currentKey] = []

        else:
            pid = float(res[0])
            dist = float(res[1])
            meanReal += dist
            queryToResReal[currentKey].append((pid,dist))
            
    meanReal = meanReal / n

    f = open(results)

    meanTest = 0
    nTest = f.readline()
    kTest = f.readline()
    for x in f:
        res = x.split()
        if(res[0].startswith('q')):
            currentKey = int(res[1])
            queryToResTest[currentKey] = []
        else:
            pid = float(res[0])
            dist = float(res[1])
            meanTest += dist
            queryToResTest[currentKey].append((pid,dist))
            
    meanTest = meanTest / n
            
    avgRecall = float(0)
    avgMSE = float(0)
    varianceReal = 0
    varianceTest = 0
    query = 0
    results = []
    for i in range(0,len(queryToResReal)):
        recall = 0 
        for j in queryToResTest[i]:
            for l in queryToResReal[i]: 
                if j[0] == l[0]:
                    recall += 1
        recall = recall / k
        print("Query " + str(i) + " has recall: " + str(recall))

        results.append((query,recall))
        query += 1
        avgRecall += recall

    avgRecall = avgRecall / n
    avgMSE = avgMSE / n
    varianceReal = varianceReal / n
    varianceTest = varianceTest / n

    print("Average recall: " + str(avgRecall))


if __name__ == "__main__":
    main()