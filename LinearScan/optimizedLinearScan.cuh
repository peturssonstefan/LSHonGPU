#pragma once 
#include "point.h"
#include "resultDTO.h"

Result runOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries);