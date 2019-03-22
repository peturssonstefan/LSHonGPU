#pragma once

#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"


__inline__ __device__ 
float generalizedJaccardDistance(float* data, float* query, int dimensions) 
{
	float minv = 0; 
	float maxv = 0;
	for(int i = 0; i < dimensions; i++) {
		minv += min(data[i], query[i]); 
		minv += max(data[i], query[i]);
	}

	return minv / maxv; 
}