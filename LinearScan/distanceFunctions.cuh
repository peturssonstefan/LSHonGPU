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

template<class T> __inline__ __device__
float hammingDistanceFunc(T* data, T* query, int sketchDim) {
	float hammingDistance = 0;
	for (int j = 0; j < sketchDim; j++) {
		unsigned int queryValue = query[j];
		unsigned int dataValue = data[j];
		unsigned int bits = queryValue ^ dataValue;
		int bitCount = __popc(bits);
		hammingDistance += bitCount;
	}

	return hammingDistance;
}

template<class T> __inline__ __device__
float euclideanDistanceFunc(T* data, T* query, int sketchDim) {
	float distance = 0;
	
	for (int j = 0; j < sketchDim; j++) {
		float queryVal = query[j];
		float dataVal = data[j];
		float dist = pow((queryVal - dataVal), 2);
		distance += dist;
	}

	return distance;
}


template<class T> __inline__ __device__
float jaccardDistanceFunc(T* data, T* query, int sketchDim, int similarityDivisor) {
	float jaccardSimilarity = 0;

	for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {
		T dataSketch = data[hashIdx];
		T querySketch = query[hashIdx];
		jaccardSimilarity += dataSketch == querySketch ? 1 : 0;
	}

	jaccardSimilarity /= similarityDivisor;

	float jaccardDistance = 1 - jaccardSimilarity;
	return jaccardDistance;
}

template<class T> __inline__ __device__
float jaccardOneBitDistanceFunc(T* data, T* query, int sketchDim, int similarityDivisor) {
	float jaccardSimilarity = 0;

	for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {
		unsigned int dataSketch = data[hashIdx];
		unsigned int querySketch = query[hashIdx];
		int diff = (SKETCH_COMP_SIZE - __popc(dataSketch ^ querySketch));
		jaccardSimilarity += diff;
	}

	jaccardSimilarity /= similarityDivisor;

	float jaccardDistance = 1 - jaccardSimilarity;

	return jaccardDistance;
}

template<class T> __inline__ __device__
float runSketchedDistanceFunction(int implementation, T* data, T* query, int sketchDim, int similarityDivisor = 1) {
	switch (implementation)
	{
	case 3:
		return hammingDistanceFunc(data, query, sketchDim);
		break;
	case 4: 
		return jaccardDistanceFunc(data, query, sketchDim, similarityDivisor);
		break;
	case 5: 
		return jaccardOneBitDistanceFunc(data, query, sketchDim, similarityDivisor);
		break;
	case 6:
		return euclideanDistanceFunc(data, query, sketchDim);
		break;
	default:
		break;
	}
}