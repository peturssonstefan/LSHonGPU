#pragma once
#include "point.h"

Point* runWeightedMinHash(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int implementation);