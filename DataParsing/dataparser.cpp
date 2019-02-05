#include<iostream>
#include"Models/point.h"
#include"Parsers/gloveparser.h"
#include"settings.h"
#include<string>
using namespace std;

int main(int argc, char** args){
    Settings settings;

    settings.filename = args[1];
    settings.N = atoi(args[2]);
    settings.D = atoi(args[3]);

    //TODO make if statements for the different parses needed
    GloveParser parser;
    vector<Point> data = parser.parse(settings);


    for(Point p : data){
        cout << p.ID << ": ";
        for(float x : p.data){
            cout << x << " ";
        }
        cout << endl;
    }
}