#define _USE_MATH_DEFINES

#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<numeric>
#include<limits>
#include<functional>
#include<cmath>

using namespace std;

struct goldstandard
{
    int N;
    int dimensions;
    int k;
    vector<vector<float>> data;
    vector<vector<float>> queries;
};


vector<vector<float>> readData(string filename, int dimensions, int N){
    ifstream file;

    file.open(filename);

    if(!file){
        cout << "File was not able to be opened" << endl;
        exit(1);
    }

    vector<vector<float>> data;

    for(int j = 0; j < N; j++){
        vector<float> point(dimensions);
        float x;
        for(int i = 0; i < dimensions; i++){
            file >> x;
            point[i] = x;
        }

        data.push_back(point);
    }

    return data;
}

void printData(vector<vector<float>> data){
    for(int i = 0; i < data.size(); i++)
    {
        for(int j = 0; j < data[0].size(); j++)
        {
            cout << data[i][j] << " ";
        }
        cout << endl;
        
    }
}

void printPoint(vector<float> point){
    for(float x : point){
        cout << x << " ";
    }
    cout << endl;
}

float magnitude(vector<float> x){
    return sqrt(accumulate(x.begin(), x.end(), 0.0, [](float state,float xi){ return state + (xi*xi); }));
}

float cosineDistance(vector<float> x, vector<float> y){
    float dotProd = inner_product(x.begin(), x.end(), y.begin(), 0.0);

    float xMagnitude = magnitude(x);
    float yMagnitude = magnitude(y);

    float sim = dotProd/(xMagnitude * yMagnitude);
    
    float angularDistance = acos(sim)/M_PI;

    return angularDistance;
}

void computeGoldStandard(goldstandard settings){
    for(vector<float> q : settings.queries){
        vector<float> nearest;
        float minDist = numeric_limits<float>::infinity();
        for(vector<float> x : settings.data){
            float dist = cosineDistance(q,x);
            cout << "Dist calculated: " << dist << endl;
            printPoint(q);
            printPoint(x);
            if(dist < minDist){
                minDist = dist;
                nearest = x;
            }
        }

        cout << "Nearest" << endl;
        printPoint(q);
        printPoint(nearest);
    }


}

int main(int argc, char** args){
    
    goldstandard settings;

    settings.N = stoi(args[2]);
    settings.dimensions = stoi(args[3]);
    settings.k = stoi(args[4]);

    settings.data = readData(args[1], settings.dimensions, settings.N);
    printData(settings.data);

    settings.queries.push_back(settings.data[settings.data.size()-1]);
    settings.data.pop_back();

    computeGoldStandard(settings);

    return 0;
}