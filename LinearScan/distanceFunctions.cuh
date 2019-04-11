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
	for (int i = 0; i < dimensions; i++) {
		minv += min(data[i], query[i]);
		maxv += max(data[i], query[i]);
	}

	return 1-(minv / maxv);
}

__inline__ __device__ 
float angularDistance(float* data, float* query, int dimensions, float magnitude_query) {
	float dotProduct = 0; 
	float magnitude_data = 0.0;
	float angular_distance = 0.0; 


	for (int j = 0; j < dimensions; j++) {
		dotProduct += query[j] * data[j];
		magnitude_data += data[j] * data[j];
	}

	magnitude_data = sqrt(magnitude_data);

	angular_distance = 1 -(dotProduct / (magnitude_query * magnitude_data));
	return angular_distance;
}

__inline__ __device__
float euclideanDistance(float* data, float* query, int dimensions) {
	float euclideanDist= 0;

	for (int j = 0; j < dimensions; j++) {
		euclideanDist += pow((query[j] - data[j]), 2);
	}

	euclideanDist = sqrt(euclideanDist);
	return euclideanDist;
}

__inline__ __device__
float runDistanceFunction(int func, float* data, float* query, int dimensions, float magnitude_query) {
	float distance = 0; 
	switch (func) {
	case 1:
		distance = angularDistance(data, query, dimensions, magnitude_query);
		break;
	case 2:
		distance = generalizedJaccardDistance(data, query, dimensions);
		break;
	case 3: 
		distance = euclideanDistance(data, query, dimensions);
		break; 
	default: printf("Invalid operation selected for distance function \n"); return;
	}

	return distance; 
}