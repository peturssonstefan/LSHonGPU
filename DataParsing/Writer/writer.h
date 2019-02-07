#pragma once 
#include"../Models/point.h"
#include"../settings.h"
#include<vector>

class Writer
{    
    public:
        void writeData(vector<Point> data, char* filename);
};
