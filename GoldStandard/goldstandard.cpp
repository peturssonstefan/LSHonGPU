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
    vector<point> data;
    vector<point> queries;
    string outFile;
};


vector<point> readData(string filename){
    ifstream file;

    file.open(filename);

    if(!file){
        cout << "File was not able to be opened" << endl;
        exit(1);
    }

    int N;
    int dimensions;

    file >> N;
    file >> dimensions;

    vector<point> data;

    for(int j = 0; j < N; j++){
        vector<float> elements(dimensions);
        int ID;
        file >> ID;

        float x;
        for(int i = 0; i < dimensions; i++){
            file >> x;
            elements[i] = x;
        }

        point p;

        p.elements = elements;
        p.id = ID;

        data.push_back(p);
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

    cout << xMagnitude << " - " << yMagnitude << endl;

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

    writeDataInfo(file, settings.queries.size(), settings.k);
    int processedQueries = 0;
    for(point q : settings.queries){
        // use PQ for storing k min elements
        auto compare = [](point x, point y) { return x.cachedDistance < y.cachedDistance; };
        priority_queue<point, vector<point>, decltype(compare)> queue(compare);

        for(point x : settings.data){
            x.cachedDistance = innerProduct(q,x);
            //cout << "Dist calculated: " << q.id << " - " << x.id << " " << x.cachedDistance << endl;

            queue.push(x);

            if(queue.size() > settings.k){
                queue.pop();
            }
        }

        writeQueryInfo(file, q);

        // reverse queue to get smallet distance first
        vector<point> neihbors(queue.size());
        for(int i = queue.size()-1 ; i >= 0; i--){
            neihbors[i] = queue.top();
            queue.pop();
        }

        for(point p : neihbors){
            writeNeighborInfo(file, p);
        }

        cout << "Processed query: " << processedQueries++ + 1 << " of " << settings.queries.size() << endl;
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

    settings.data = readData(args[2]);
    settings.queries = readData(args[3]);

    computeGoldStandard(settings);

    return 0;
}