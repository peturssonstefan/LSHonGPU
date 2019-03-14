#include "sortingFramework.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"
#include "candidateSetScanner.cuh"

__inline__ __device__
void scanHammingDistance(float* originalData, float* originalQuery, int dimensions, unsigned long * data, unsigned long * queries, int sketchDim, int N_data, int N_query, int k, Point* result)
{
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE; 
	Parameters params; 
	params.lane = lane; 
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;
	int maxKDistance = 0;
	int warpQueueSize = k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;


//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, SKETCH_COMP_SIZE * sketchDim + 1);
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

		Point currentPoint = createPoint(i, (float)hammingDistance); 

		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance)) {
			for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
				for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
					if (threadQueue[i].distance < threadQueue[j].distance) {
						swapPoint = threadQueue[j];
						threadQueue[j] = threadQueue[i];
						threadQueue[i] = swapPoint;
					}
				}
			}
			laneStrideSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
		}

	}
	
	//Sort before candidateSetScan if we only do exact calculations on warp queue elements.

	//Candidate set scan.
	candidateSetScan(originalData, originalQuery, dimensions, threadQueue, k);

	startSort(threadQueue, swapPoint, params);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;

	for (int i = kIdx; i < k; i += WARPSIZE)
	{
		result[resultIdx + i] = threadQueue[warpQueueIdx--];
	}

}
