#include "gloveRunner.h"
#include<iostream>
#include"../settings.h"
#include"../Parsers/gloveparser.h"
#include"../Models/point.h"
#include<random>
#include"../Writer/writer.h"

using namespace std;

void GloveRunner::run(int argc, char** args){
    if(argc != 8){
        cout << "Wrong number of parameters - program needs following parameters: " << endl;
        cout << "glove InputFile N D NumberOfQueryPoints OutFileData OutFileQueries" << endl;
        return;
    }
    
    // Read in settings to run
    Settings settings;

    settings.filename = args[2];
    settings.N = atoi(args[3]);
    settings.D = atoi(args[4]);
    settings.NumQueryPoints = atoi(args[5]);

    settings.outFileData = args[6];
    settings.outFileQueries = args[7];

    // Parse data
    GloveParser parser;
    vector<Point> data = parser.parse(settings);

    // Divide data into data and query points
    vector<Point> queryPoints(settings.NumQueryPoints);

    // Setup for getting random indexes
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, settings.N); // define the range

    std::default_random_engine generator;

    for(int i = 0; i < settings.NumQueryPoints; i++){
        int randomI = distr(generator);
        // Avoid index out of bounds after query points are taken from data
        while (randomI >= data.size()){
            randomI = distr(generator);
        }

        queryPoints[i] = data[randomI];
        data.erase(data.begin() + randomI, data.begin() + randomI + 1);
        cout << "Found query point:" << i + 1 << " of " << settings.NumQueryPoints << endl;
    }


    // Write data and query points to file
    Writer writer;
    writer.writeData(data, settings.outFileData);
    writer.writeData(queryPoints, settings.outFileQueries);
}