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

    vector<Point> data = parser.parse(inputFileData);
    vector<Point> queires = parser.parse(inputFileQueries);

    // Write formated
    Writer writer;
    writer.writeData(data, outputFileData);
    writer.writeData(queires, outputFileQueries);
}