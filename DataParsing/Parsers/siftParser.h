#pragma once
#include<vector>
#include"../Models/point.h"

class SiftParser
{
public:
    vector<Point> parse(char* fileName);
};
