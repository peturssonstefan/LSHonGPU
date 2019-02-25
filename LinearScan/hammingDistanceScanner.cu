#include "hammingDistanceScanner.cuh"
#include "shuffleUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"

__device__
void scanHammingDistance(long * data, long * queries, int sketchDim, int N_data, int N_query, int k, Point* threadQueue, Point* result)
{
	Point candidateItems[MAX_K];

#pragma unroll
	for (int i = 0; i < MAX_K; i++) {
		Point p;
		p.distance = SKETCH_COMP_SIZE * sketchDim + 1; 
		p.ID = -1;
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int queryIndex = blockIdx.x * sketchDim;
	int tIndex = threadId * k;

	for (int i = threadIdx.x; i < N_data; i += blockDim.x) {
		int hammingDistance = 0; 

#pragma unroll
		for (int j = 0; j < sketchDim; j++) {
			unsigned long bits = queries[queryIndex + j] ^ data[sketchDim*i + j];
			printf("bits: %lu \n", bits); 
			hammingDistance += __popcll(bits);
		}

		Point currentPoint;

		currentPoint.ID = i;
		currentPoint.distance = hammingDistance;

		Point swapPoint;

		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (candidateItems[j].distance > currentPoint.distance) {
				swapPoint = candidateItems[j];
				candidateItems[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}


	}

#pragma unroll
	for (int i = 0; i < k; i++) {
		threadQueue[tIndex + i] = candidateItems[i];
	}

	Point* kNearest = blockReduce(&threadQueue[tIndex], k);

	if (threadIdx.x == 0) {

#pragma unroll
		for (int i = 0; i < k; i++) {
			result[blockIdx.x * k + i] = kNearest[i];
		}
	}
}
