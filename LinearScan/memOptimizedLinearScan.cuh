#pragma once
#include "point.h"
#include "constants.cuh"
#include "sortParameters.h"
#include "sortingFramework.cuh"
#include "launchHelper.cuh"
#include "processingUtils.cuh"
#include "distanceFunctions.cuh"
#include "cudaHelpers.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "resultDTO.h"

Result runMemOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries, int distanceFunc);
