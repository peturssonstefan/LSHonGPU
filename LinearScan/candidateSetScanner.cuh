#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"
#include "distanceFunctions.cuh"

__inline__ __device__
void candidateSetScan(float* data, float* query, int dimensions, Point* candidates, int k, int distFunc) {
	
	float magnitude_query = 0;

	for (int j = 0; j < dimensions; j++) {
		magnitude_query += query[j] * query[j];
	}

	magnitude_query = sqrt(magnitude_query);

	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {

		int index = candidates[i].ID * dimensions; 
		float distance = 0.0; // reset value.

		distance = runDistanceFunction(distFunc, &data[index], query, dimensions, magnitude_query);

		candidates[i].distance = candidates[i].ID < 0 ? (float)INT_MAX : distance;
	}
}