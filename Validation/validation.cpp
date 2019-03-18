#include <iostream>
#include <set>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;

struct Point{
    int ID;
    float distance;
};

class PointValidation
{
    public:
        int ID;
        float Distance;

        //Overload < operator
        bool operator<(const PointValidation& other) const {
            return this->ID < other.ID;
        }

        void setFieldsFromPoint(Point p){
            this->ID = p.ID;
            this->Distance = p.distance;
        }

};

class QueryResult{
    
    public:
        string queryId;
        set<PointValidation> NN;
};


vector<QueryResult> readData(Point* data, int N_queries ,int k, int reportK){
    vector<QueryResult> parsedData(N_queries); 
    for(int queryID = 0; queryID < N_queries; queryID++){
        QueryResult qRes;
        qRes.queryId = to_string(queryID) + ":";
        for(int i = 0; i < reportK; i++){
            PointValidation pVal;

            pVal.setFieldsFromPoint(data[queryID * k + i]);

            qRes.NN.insert(pVal);
        }

        parsedData[queryID] = qRes;
    }

    return parsedData;
}

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
        set<PointValidation> pointValidationSet;        
        for(int resultNum = 0; resultNum < k; resultNum++){
            PointValidation p;

            int pointValidationId = 0;
            float pointValidationDistance = 0.0;

            file >> pointValidationId;
            file >> pointValidationDistance;

            p.ID = pointValidationId;
            p.Distance = pointValidationDistance;

            pointValidationSet.insert(p);
        }

        QueryResult result;
        result.queryId = queryId;
        result.NN = pointValidationSet;

        results[queryNum] = result;
    }
    
    return results;
}

void printData(vector<QueryResult> data){    
    for(int i = 0; i < data.size(); i++){
        set<PointValidation> pSet = data[i].NN;
        
        cout << data[i].queryId << endl;

        // Creating a iterator pointValidationing to start of set
        set<PointValidation>::iterator it = pSet.begin();
        // Iterate till the end of set
        while (it != pSet.end())
        {   
            PointValidation p = (*it);
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

        // Creating a iterator pointValidationing to start of set
        set<PointValidation>::iterator it = truth.NN.begin();

        set<PointValidation>::iterator findIt;

        float recalledElements = 0;

        // Iterate till the end of set
        while (it != truth.NN.end())
        {   
            PointValidation p = (*it);
            
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

// Function for calling into the framework from KNN framework
void runValidation(Point* truths, Point* results, int N_queries, int k, int reportK){

    vector<QueryResult> truthsVal = readData(truths, N_queries, k, reportK);
    vector<QueryResult> resultsVal = readData(results, N_queries, k, reportK);

    calculateRecall(truthsVal, resultsVal);
}

int main(int argc, char** args){
    char* truthFile = args[1];
    char* resultFile = args[2];

    vector<QueryResult> truth = readData(truthFile);
    vector<QueryResult> results = readData(resultFile);

    calculateRecall(truth, results);

    return 0;
}