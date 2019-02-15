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

    outFile << data.size() << endl;
    outFile << data[0].data.size() << endl;

    int ID = 0;
    for(Point p : data){
        outFile << ID++ << " ";

        for(float x : p.data){
            outFile << x << " ";
        }
        outFile << endl;
    }    

    outFile.close();
}