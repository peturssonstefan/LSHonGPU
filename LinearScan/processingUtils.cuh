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
			atomicMin(&minValues[dim], (int)data[i * dimensions + dim]); // floor by casting
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMin(&minValues[dim], (int)queries[i * dimensions + dim]); // floor by casting
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
