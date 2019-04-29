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
        vector<PointValidation> truthVector;
};


vector<QueryResult> readData(Point* data, int N_queries ,int k, int reportK, bool isTruthFile){
    vector<QueryResult> parsedData(N_queries); 
    for(int queryID = 0; queryID < N_queries; queryID++){
        QueryResult qRes;
        qRes.queryId = to_string(queryID) + ":";
        for(int i = 0; i < reportK; i++){
            PointValidation pVal;

            pVal.setFieldsFromPoint(data[queryID * k + i]);

            if(isTruthFile){
                qRes.truthVector.push_back(pVal);
            } else {
                qRes.NN.insert(pVal);
            }
        }

        parsedData[queryID] = qRes;
    }

    return parsedData;
}

vector<QueryResult> readData(char* filename, bool isTruthFile){
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

        QueryResult result;
        result.queryId = queryId;

        for(int resultNum = 0; resultNum < k; resultNum++){
            PointValidation p;

            int pointValidationId = 0;
            float pointValidationDistance = 0.0;

            file >> pointValidationId;
            file >> pointValidationDistance;

            p.ID = pointValidationId;
            p.Distance = pointValidationDistance;

            if(isTruthFile){
                result.truthVector.push_back(p);
            } else {
                result.NN.insert(p);
            }

        }

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

float calculateRecall(vector<QueryResult> truths, vector<QueryResult> results, int k){
    float totalRecall = 0;
    for(int queryNum = 0; queryNum < truths.size(); queryNum++){
        QueryResult result = results[queryNum];
        QueryResult truth = truths[queryNum];

        set<PointValidation>::iterator findIt;

        float recalledElements = 0;

        // Iterate till the end of set
        for(int i = 0; i < truth.truthVector.size() && i < k; i++)
        {   
            PointValidation p = truth.truthVector[i];
            
            findIt = result.NN.find(p);

            if(findIt != result.NN.end()){
                recalledElements++;
            } else {
                cout << "Query: " << truth.queryId << " did not find: " << p.ID << endl;
            }
        }

        float recall = recalledElements/k;
        
        totalRecall += recall;
    }

    cout << "Averge recall: " << totalRecall / truths.size() << endl;
	return  totalRecall / truths.size();
}

float calculateDistanceRatio(vector<QueryResult> truths, vector<QueryResult> results, int k) {
	
	float totalAverage = 0;
	for (int queryNum = 0; queryNum < truths.size(); queryNum++) {
		QueryResult result = results[queryNum];
		QueryResult truth = truths[queryNum];

		float truthsDistance = 0;
		float resultsDistance = 0;

        // calculate truths distance
        for(int i = 0; i < truth.truthVector.size() && i < k; i++){
            PointValidation p = truth.truthVector[i];

            truthsDistance += p.Distance;
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
	return totalAverage / truths.size(); 
}

// Function for calling into the framework from KNN framework
void runValidation(char* truths, float* container, Point* results, int N_queries, int k, int reportK){
    vector<QueryResult> truthsVal = readData(truths, true);
    vector<QueryResult> resultsVal = readData(results, N_queries, k, reportK, false);
    float recall = calculateRecall(truthsVal, resultsVal,reportK);
	float avgDistance = calculateDistanceRatio(truthsVal, resultsVal, reportK);
	container[0] = recall; 
	container[1] = avgDistance; 
}

bool compareDistance(PointValidation p1, PointValidation p2) {
	return p1.Distance < p2.Distance; 
}

float calculateAvgDistanceGivenK(vector<QueryResult> truths, vector<QueryResult> points, int k) {
	float totalAvgDistance = 0; 
	for (int i = 0; i < truths.size(); i++) {
		QueryResult truth = truths[i];
		QueryResult point = points[i];
		float truthsDistance = 0;
		float resultsDistance = 0;

		vector<PointValidation> truthNNvector(truth.NN.size());
		vector<PointValidation> pointNNvector(point.NN.size());
		copy(truth.NN.begin(), truth.NN.end(), truthNNvector.begin());
		copy(point.NN.begin(), point.NN.end(), pointNNvector.begin());
		sort(truthNNvector.begin(), truthNNvector.end(), compareDistance);

		for (int i = 0; i < k; i++) {
			truthsDistance += truthNNvector[i].Distance; 
		}
		for (int i = 0; i < pointNNvector.size(); i++) {
			resultsDistance += pointNNvector[i].Distance;
		}

		float distanceRatio = resultsDistance / truthsDistance;
		totalAvgDistance += distanceRatio; 
	}

	return totalAvgDistance / truths.size();
}

float calculateRecallGivenK(vector<QueryResult> truths, vector<QueryResult> points, int k) {

	float totalRecall = 0;
	for (int i = 0; i < truths.size(); i++) {
		QueryResult truth = truths[i];
		QueryResult point = points[i];

		vector<PointValidation> truthNNvector(truth.NN.size()); 
		vector<PointValidation> pointNNvector(point.NN.size()); 
		copy(truth.NN.begin(), truth.NN.end(), truthNNvector.begin());
		copy(point.NN.begin(), point.NN.end(), pointNNvector.begin());

		sort(truthNNvector.begin(), truthNNvector.end(), compareDistance); 

		for (int i = 0; i < pointNNvector.size(); i++) {
			PointValidation p = pointNNvector[i]; 
			for (int i = 0; i < k; i++) {
				PointValidation tp = truthNNvector[i]; 
				if (tp.ID == p.ID) totalRecall++;
			}
		}
	}

	return totalRecall / (points.size() * k); 
}


void runValidationFromLargeFile(char* truths, float* container, Point* results, int N_queries, int k, int reportK) {
	vector<QueryResult> truthsVal = readData(truths, true);
	vector<QueryResult> resultsVal = readData(results, N_queries, k, reportK, false);
	float recall = calculateRecallGivenK(truthsVal, resultsVal, k); 
	float distance = calculateAvgDistanceGivenK(truthsVal, resultsVal, k); 
	cout << "r: " << recall << endl;
	cout << "d: " << distance << endl;
	container[0] = recall; 
	container[1] = distance; 
}

int main(int argc, char** args){
    char* truthFile = args[1];
    char* resultFile = args[2];
    int k = atoi(args[3]);

    vector<QueryResult> truth = readData(truthFile, true);
    vector<QueryResult> results = readData(resultFile, false);

    //printData(truth);

    calculateRecall(truth, results, k);
    calculateDistanceRatio(truth, results, k);

    return 0;
}