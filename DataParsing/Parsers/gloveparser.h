#include<iostream>
#include"../Models/point.h"
#include<vector>
#include"../settings.h"

class GloveParser{
    public:
        vector<Point> parse(Settings settings);
};