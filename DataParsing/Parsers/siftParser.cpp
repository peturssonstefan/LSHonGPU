#include<iostream>
#include"siftParser.h"
#include"../Models/point.h"
#include<vector>

using namespace std;

vector<Point> SiftParser::parse(char* fileName){
    FILE *input = fopen(fileName, "rb");
    vector<Point> dataset;
    while(true) {
        int d;
        if (!fread(&d, sizeof(int), 1, input)) { // read until there are no more lines to read
            break;
        }

        vector<float> current_row(d);
        fread(&current_row[0], sizeof(float), d, input);
        Point p;
        p.data = current_row;
        
        dataset.push_back(p);
    }

    fclose(input);

    return dataset;
}
