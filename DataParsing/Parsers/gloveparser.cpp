#include<iostream>
#include<vector>
#include"gloveparser.h"
#include<fstream>
#include<string>

using namespace std;

vector<Point> GloveParser::parse(Settings settings){
    ifstream file;

    file.open(settings.filename);

    if(!file){
        cout << "File was not able to be opened" << endl;
        exit(1);
    }

    vector<Point> data;

    for(int i = 0; i < settings.N; i++){
        Point p;
        string word;
        file >> word;

        float x;
        for(int j = 0; j < settings.D; j++){
            file >> x;
            p.data.push_back(x);
        }

        data.push_back(p);
    }

    file.close();

    return data; 
}