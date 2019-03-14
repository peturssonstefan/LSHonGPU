#include "shuffleUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"

__inline__ __device__
void scanHammingDistance(unsigned long * data, unsigned long * queries, int sketchDim, int N_data, int N_query, int k, Point* threadQueue, Point* result)
{
	Point candidateItems[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE; 
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;

//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		candidateItems[i] = createPoint(-1, SKETCH_COMP_SIZE * sketchDim + 1);
	}

	for (int i = lane; i < N_data; i += WARPSIZE) {
		int hammingDistance = 0;

#pragma unroll
		for (int j = 0; j < sketchDim; j++) {
			unsigned long queryValue = queries[queryIdx + j];
			unsigned long dataValue = data[sketchDim*i + j]; 
			unsigned long bits = queryValue ^ dataValue; 
			int bitCount = __popcll(bits);
			hammingDistance += bitCount;
		}

		Point currentPoint = createPoint(i, hammingDistance); 
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
