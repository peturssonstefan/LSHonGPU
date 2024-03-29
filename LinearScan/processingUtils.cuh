#pragma once

#include <cuda_runtime.h>
#include "point.h"
#include "constants.cuh"
#include "pointExtensions.cuh"
#include "sortParameters.h"
#include <math.h>
#include <cuda.h>


__inline__ __device__
void transformData(float* data, float* queries, int N_data, int N_queries, int dimensions, int* minValues) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;

	// Find min
	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMin(&minValues[dim], floor(data[i * dimensions + dim])); 
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMin(&minValues[dim], floor(queries[i * dimensions + dim])); // floor by casting
		}
	}

	__syncthreads();

	// Transform data
	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			data[i * dimensions + dim] += abs(minValues[dim]);
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			queries[i * dimensions + dim] += abs(minValues[dim]);
		}
	}
	
}

__inline__ __device__
void transformToUnitVectors(float* data, int N_data, int dimensions) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadId; i < N_data; i += gridDim.x * blockDim.x) {
		float magnitude = 0;
		
		int pointIndex = i * dimensions;

		for (int dim = 0; dim < dimensions; dim++) {
			magnitude += data[pointIndex + dim] * data[pointIndex + dim];
		}

		magnitude = sqrt(magnitude);

		for (int dim = 0; dim < dimensions; dim++) {
			data[pointIndex + dim] = data[pointIndex + dim] / magnitude;
		}
	}
}

