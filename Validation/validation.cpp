#include <iostream>
#include <set>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;

class Point
{
    public:
        int ID;
        float Distance;

        //Overload < operator
        bool operator<(const Point& other) const {
            return this->ID < other.ID;
        }

};

class QueryResult{
    
    public:
        string queryId;
        set<Point> NN;
};


vector<QueryResult> readData(char* filename){
    ifstream file;
    file.open(filename);

    int n = 0;
    int k = 0;

    file >> n;
    file >> k;

    vector<QueryResult> results(n);

    string queryId;
    for(int queryNum = 0; queryNum < n; queryNum++){
        file >> queryId;
        set<Point> pointSet;        
        for(int resultNum = 0; resultNum < k; resultNum++){
            Point p;

            int pointId = 0;
            float pointDistance = 0.0;

            file >> pointId;
            file >> pointDistance;

            p.ID = pointId;
            p.Distance = pointDistance;

            pointSet.insert(p);
        }

        QueryResult result;
        result.queryId = queryId;
        result.NN = pointSet;

        results[queryNum] = result;
    }
    
    return results;
}

void printData(vector<QueryResult> data){    
    for(int i = 0; i < data.size(); i++){
        set<Point> pSet = data[i].NN;
        
        cout << data[i].queryId << endl;

        // Creating a iterator pointing to start of set
        set<Point>::iterator it = pSet.begin();
        // Iterate till the end of set
        while (it != pSet.end())
        {   
            Point p = (*it);
            // Print the element
            cout << p.ID << " " << p.Distance << endl;;
            //Increment the iterator
            it++;
        }
    }
}

void calculateRecall(vector<QueryResult> truths, vector<QueryResult> results){
    float totalRecall = 0;
    for(int queryNum = 0; queryNum < truths.size(); queryNum++){
        QueryResult result = results[queryNum];
        QueryResult truth = truths[queryNum];

        // Creating a iterator pointing to start of set
        set<Point>::iterator it = truth.NN.begin();

        set<Point>::iterator findIt;

        float recalledElements = 0;

        // Iterate till the end of set
        while (it != truth.NN.end())
        {   
            Point p = (*it);
            
            findIt = result.NN.find(p);

            if(findIt != result.NN.end()){
                recalledElements++;
            }

            //Increment the iterator
            it++;
        }

        float recall = recalledElements/truth.NN.size();
        cout << "recall for " << truth.queryId << " = " << recall << endl;
        totalRecall += recall;
    }

    cout << "Averge recall: " << totalRecall / truths.size() << endl;
}

int main(int argc, char** args){
    char* truthFile = args[1];
    char* resultFile = args[2];

    vector<QueryResult> truth = readData(truthFile);
    vector<QueryResult> results = readData(resultFile);

    calculateRecall(truth, results);

    return 0;
}