#include<iostream>
#include"siftRunner.h"
#include"../Parsers/siftParser.h"
#include"../Models/point.h"
#include"../Writer/writer.h"

using namespace std;

void SiftRunner::run(int argc, char** args){
    if(argc != 6){
        cout << "Not the right amount of arguments for sift" << endl;
        cout << "sift InputDataFile InputQueriesFile OutputDataFile OutputQueriesFile" << endl;
        return;
    }
    
    char* inputFileData = args[2];
    char* inputFileQueries = args[3];
    char* outputFileData = args[4];
    char* outputFileQueries = args[5];

    SiftParser parser;

    int id;

    vector<Point> data = parser.parse(inputFileData, &id);
    vector<Point> queires = parser.parse(inputFileQueries, &id);

    cout << id << endl;

    // Write formated
    Writer writer;
    writer.writeData(data, outputFileData);
    writer.writeData(queires, outputFileQueries);
}