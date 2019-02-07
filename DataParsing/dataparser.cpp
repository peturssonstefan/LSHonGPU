#include<iostream>
#include"Models/point.h"
#include"Parsers/gloveparser.h"
#include"settings.h"
#include<string>
#include"Writer/writer.h"

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

    // Write data to file
    Writer writer;
    writer.writeData(data, settings.outFileData);


    // for(Point p : data){
    //     cout << p.ID << ": ";
    //     for(float x : p.data){
    //         cout << x << " ";
    //     }
    //     cout << endl;
    // }

    return 0;
}