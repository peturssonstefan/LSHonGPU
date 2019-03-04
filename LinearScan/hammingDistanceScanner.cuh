#include "shuffleUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"

__inline__ __device__
void scanHammingDistance(unsigned long * data, unsigned long * queries, int sketchDim, int N_data, int N_query, int k, Point* threadQueue, Point* result)
{
	Point candidateItems[MAX_K];

#pragma unroll
	for (int i = 0; i < MAX_K; i++) {
		Point p;
		p.distance = SKETCH_COMP_SIZE * sketchDim + 1;
		p.ID = -1;
		candidateItems[i] = p;
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int queryIndex = blockIdx.x * sketchDim;
	int tIndex = threadId * k;
	for (int i = threadIdx.x; i < N_data; i += blockDim.x) {
		int hammingDistance = 0;

#pragma unroll
		for (int j = 0; j < sketchDim; j++) {
			unsigned long queryValue = queries[queryIndex + j]; 
			unsigned long dataValue = data[sketchDim*i + j]; 
			unsigned long bits = queryValue ^ dataValue; 
			int bitCount = __popcll(bits);
			hammingDistance += bitCount;
		}

		Point currentPoint;

		currentPoint.ID = i;
		currentPoint.distance = hammingDistance;

		Point swapPoint;

		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (currentPoint.distance < candidateItems[j].distance) {
				swapPoint = candidateItems[j];
				candidateItems[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

	}

#pragma unroll
	for (int i = 0; i < k; i++) {
		threadQueue[i] = candidateItems[i];
	}

	Point* kNearest = blockReduce(threadQueue, k, SKETCH_COMP_SIZE*sketchDim);
	if (threadIdx.x == 0) {
#pragma unroll
		for (int i = 0; i < k; i++) {
			result[blockIdx.x * k + i] = kNearest[i];
		}
	}
}
