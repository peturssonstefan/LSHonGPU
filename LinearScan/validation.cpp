#include <iostream>
#include <set>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include "point.h"

using namespace std;

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
        //cout << "recall for " << truth.queryId << " = " << recall << endl;
        totalRecall += recall;
    }

    cout << "Averge recall: " << totalRecall / truths.size() << endl;
}

void calculateDistanceRatio(vector<QueryResult> truths, vector<QueryResult> results) {
	
	float totalAverage = 0;
	for (int queryNum = 0; queryNum < truths.size(); queryNum++) {
		QueryResult result = results[queryNum];
		QueryResult truth = truths[queryNum];

		float truthsDistance = 0;
		float resultsDistance = 0;

		// Creating a iterator pointValidationing to start of set
		set<PointValidation>::iterator it = truth.NN.begin();

		// Iterate till the end of set
		while (it != truth.NN.end())
		{
			PointValidation p = (*it);

			truthsDistance += p.Distance;

			//Increment the iterator
			it++;
		}

		set<PointValidation>::iterator resIt = result.NN.begin();

		// Iterate till the end of set
		while (resIt != result.NN.end())
		{
			PointValidation p = (*resIt);

			resultsDistance += p.Distance;

			//Increment the iterator
			resIt++;
		}

		float distanceRatio = resultsDistance / truthsDistance;
		//printf("Query: %s = %f resdist: %f truthsDist: %f\n", truth.queryId, distanceRatio, resultsDistance, truthsDistance);

		totalAverage += distanceRatio;
	}

	cout << "Total average: " << totalAverage / truths.size() << endl;

}

// Function for calling into the framework from KNN framework
void runValidation(char* truths, Point* results, int N_queries, int k, int reportK){
    vector<QueryResult> truthsVal = readData(truths);
    vector<QueryResult> resultsVal = readData(results, N_queries, k, reportK);

    calculateRecall(truthsVal, resultsVal);
	calculateDistanceRatio(truthsVal, resultsVal);
}
