#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"

__inline__ __device__
void candidateSetScan(float* data, float* query, int dimensions, Point* candidates, int k) {
	
	float magnitude_query = 0;

	for (int j = 0; j < dimensions; j++) {
		magnitude_query += query[j] * query[j];
	}

	magnitude_query = sqrt(magnitude_query);

	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		int index = candidates[i].ID * dimensions; 
		float dotProduct = 0; // reset value.
		float magnitude_data = 0.0; // reset value.
		float angular_distance = 0.0; // reset value.

		for (int j = 0; j < dimensions; j++) {
			dotProduct += query[j] * data[index + j];
			magnitude_data += data[index + j] * data[index + j];
		}

		magnitude_data = sqrt(magnitude_data);
		angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		candidates[i].distance = angular_distance; 
	}
}