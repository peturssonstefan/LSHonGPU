#include"writer.h"
#include<vector>
#include"../Models/point.h"
#include"../settings.h"
#include<fstream>
#include<stdio.h>

using namespace std;

void Writer::writeData(vector<Point> data, char* filename){
    ofstream outFile;

    outFile.open(filename);

    for(Point p : data){
        outFile << p.ID << " ";

        for(float x : p.data){
            outFile << x << " ";
        }
        outFile << endl;
    }    

    outFile.close();
}