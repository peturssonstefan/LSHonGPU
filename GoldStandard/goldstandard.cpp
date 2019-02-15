#define _USE_MATH_DEFINES

#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<numeric>
#include<limits>
#include<functional>
#include<cmath>
#include<queue>
#include<random>
#include<algorithm>

using namespace std;

struct point
{
    vector<float> elements;
    float cachedDistance;
    int id;
};

struct goldstandard
{
    int k;
    int nq;
    int nd;
    int d;
    point* data;
    point* queries;
    string outFile;
};

point* readData(string filename, int* n, int* d){
    ifstream file;

    file.open(filename);

    if(!file){
        cout << "File was not able to be opened" << endl;
        exit(1);
    }

    file >> (*n);
    file >> (*d);

    point* data = new point[(*n)];

    for(int j = 0; j < (*n); j++){
        vector<float> elements((*d));
        int ID;
        file >> ID;

        float x;
        for(int i = 0; i < (*d); i++){
            file >> x;
            elements[i] = x;
        }

        point p;

        p.elements = elements;
        p.id = ID;

        data[j] = p;
    }

    file.close();

    return data; 
}

void printData(vector<point> data){
    for(int i = 0; i < data.size(); i++)
    {
        cout << data[i].id << ": ";
        for(int j = 0; j < data[0].elements.size(); j++)
        {
            cout << data[i].elements[j] << " ";
        }
        cout << endl;
        
    }
}

void printPoint(point point){
    cout << point.id << ": ";
    // for(float x : point.elements){
    //     cout << x << " ";
    // }
    cout << endl;
}

float magnitude(point x){
    return sqrt(accumulate(x.elements.begin(), x.elements.end(), 0.0, [](float state,float xi){ return state + (xi*xi); }));
}

float cosineDistance(point x, point y){
    float dotProd = inner_product(x.elements.begin(), x.elements.end(), y.elements.begin(), 0.0);

    float xMagnitude = magnitude(x);
    float yMagnitude = magnitude(y);

    float sim = dotProd/(xMagnitude * yMagnitude);
    
    float angularDistance = acos(sim)/M_PI;

    return angularDistance;
}

float innerProduct(point x, point y){
    float dotProd = inner_product(x.elements.begin(), x.elements.end(), y.elements.begin(), 0.0);
    
    float xMagnitude = magnitude(x);
    float yMagnitude = magnitude(y);

    float sim = dotProd/(xMagnitude * yMagnitude);

    return -sim;
}

void writeDataInfo(ofstream &file, int n, int k){
    file << n << endl;
    file << k << endl;
}

void writeQueryInfo(ofstream &file, point q){
    file << q.id << ":" << endl;
}

void writeNeighborInfo(ofstream &file, point p){
    file << p.id << " " << p.cachedDistance << endl;
}

void computeGoldStandard(goldstandard settings){
    ofstream file;
    file.open(settings.outFile);

    writeDataInfo(file, settings.nq, settings.k);
    int processedQueries = 0;
    for(int qi = 0; qi < settings.nq; qi++){
        point q = settings.queries[qi];
        
        // use PQ for storing k min elements
        auto compare = [](point x, point y) { return x.cachedDistance < y.cachedDistance; };
        
        for(int pi = 0; pi < settings.nd; pi++){
            point p = settings.data[pi];
            p.cachedDistance =  innerProduct(q,p);
            //TODO fix this - probaly need pointers to the points 
            //cout << "Q: " << qi << " P: " << p.id << " - " << p.cachedDistance << endl;
            settings.data[pi] = p;
        }

        // for(int i = 0; i < settings.nd; i++){
        //     point p = settings.data[i];
        //     cout << p.cachedDistance << endl;
        // }
        
        sort(settings.data, settings.data + settings.nd, compare);

        writeQueryInfo(file, q);

        for(int i = 0; i < settings.k; i++){
            writeNeighborInfo(file, settings.data[i]);
        }

        cout << "Processed query: " << processedQueries++ + 1 << " of " << settings.nq << endl;
    }

    file.close();
}

int main(int argc, char** args){
    
    if(argc != 5){
        cout << "Not the right amount of arguments" << endl;
        cout << "K inputDataFile inputQueryFile outputFile" << endl;
        return 0;
    }

    goldstandard settings;

    settings.k = stoi(args[1]);
    settings.outFile = args[4];

    char* dataFile = args[2];
    char* queryFile = args[3];
    
    settings.data = readData(dataFile, &settings.nd, &settings.d);
    settings.queries = readData(queryFile, &settings.nq, & settings.d);

    for(int i = 0; i < settings.nq; i++){
        cout << settings.queries[i].id << endl;
    }

    computeGoldStandard(settings);

    return 0;
}