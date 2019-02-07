#include<iostream>
#include"Models/point.h"
#include"Parsers/gloveparser.h"
#include"settings.h"
#include<string>
#include"Writer/writer.h"
#include<random>

using namespace std;

int main(int argc, char** args){
    if(argc != 7){
        cout << "Wrong number of parameters - program needs following parameters: " << endl;
        cout << "InputFile N D NumberOfQueryPoints OutFileData OutFileQueries" << endl;
        return 0;
    }
    
    // Read in settings to run
    Settings settings;

    settings.filename = args[1];
    settings.N = atoi(args[2]);
    settings.D = atoi(args[3]);
    settings.NumQueryPoints = atoi(args[4]);

    settings.outFileData = args[5];
    settings.outFileQueries = args[6];

    // Parse data
    //TODO make if statements for the different parses needed
    GloveParser parser;
    vector<Point> data = parser.parse(settings);

    // Divide data into data and query points
    vector<Point> queryPoints(settings.NumQueryPoints);

    // Setup for getting random indexes
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, settings.N); // define the range

    for(int i = 0; i < settings.NumQueryPoints; i++){
        int randomI = distr(eng);
        // Avoid index out of bounds after query points are taken from data
        while (randomI >= data.size()){
            randomI = distr(eng);
        }

        queryPoints[i] = data[randomI];
        data.erase(data.begin() + randomI, data.begin() + randomI + 1);
        cout << "Found query point:" << i + 1 << " of " << settings.NumQueryPoints << endl;
    }


    // Write data and query points to file
    Writer writer;
    writer.writeData(data, settings.outFileData);
    writer.writeData(queryPoints, settings.outFileQueries);

    return 0;
}