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
    int N;
    int dimensions;
    int k;
    vector<point> data;
    vector<point> queries;
};


vector<point> readData(string filename, int dimensions, int N){
    ifstream file;

    file.open(filename);

    if(!file){
        cout << "File was not able to be opened" << endl;
        exit(1);
    }

    vector<point> data;

    for(int j = 0; j < N; j++){
        vector<float> elements(dimensions);
        string word;
        file >> word;

        float x;
        for(int i = 0; i < dimensions; i++){
            file >> x;
            elements[i] = x;
        }

        point p;

        p.elements = elements;
        p.id = j;

        data.push_back(p);
    }

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

void computeGoldStandard(goldstandard settings){
    for(point q : settings.queries){
        // use PQ for storing k min elements
        auto compare = [](point x, point y) { return x.cachedDistance < y.cachedDistance; };
        priority_queue<point, vector<point>, decltype(compare)> queue(compare);

        for(point x : settings.data){
            x.cachedDistance = cosineDistance(q,x);
            cout << "Dist calculated: " << q.id << " - " << x.id << " " << x.cachedDistance << endl;

            queue.push(x);

            if(queue.size() > settings.k){
                queue.pop();
            }
        }

        cout << "Nearest for: ";
        printPoint(q);
        while(!queue.empty()){
            cout << queue.top().id << " " << queue.top().cachedDistance << endl;
            queue.pop();
        }
    }


}

int main(int argc, char** args){
    
    goldstandard settings;

    settings.N = stoi(args[2]);
    settings.dimensions = stoi(args[3]);
    settings.k = stoi(args[4]);

    settings.data = readData(args[1], settings.dimensions, settings.N);
    printData(settings.data);


    int numQueies = 2;
    while(numQueies > 0){
        // get query points
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, settings.N); // define the range

        // get random element and delete the element from data
        int i = distr(eng);
        settings.queries.push_back(settings.data[i]);
        settings.data.erase(settings.data.begin() + i, settings.data.begin() + i + 1);

        numQueies--;
    }

    computeGoldStandard(settings);

    return 0;
}