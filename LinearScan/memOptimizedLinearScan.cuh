#pragma once
#include "point.h"
#include "constants.cuh"
#include "sortParameters.h"
#include "sortingFramework.cuh"
#include "launchHelper.cuh"
#include "processingUtils.cuh"
#include "distanceFunctions.cuh"
#include "cudaHelpers.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "resultDTO.h"

Result runMemOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries, int distanceFunc);

__inline__ __device__
void processQuery(float* queryPoint, float* dataPoints, int nData, int dimensions, int k, Point* result, int func) {

	Point threadQueue[THREAD_QUEUE_SIZE];
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int lane = threadIdx.x % WARPSIZE;
	float maxKDistance = (float)INT_MAX;
	int warpQueueSize = k / WARPSIZE;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;
	Parameters params;
	params.lane = threadIdx.x % WARPSIZE;

	int queuePosition = 0;

	//Fill thread queue with defaults
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	float magnitude_query = 0;
	float dotProduct = 0;
	float magnitude_data = 0.0;
	float distance = 0.0;


	for (int j = 0; j < dimensions; j++) {
		magnitude_query += queryPoint[j] * queryPoint[j];
	}

	magnitude_query = sqrt(magnitude_query);

	//Iterate over data; 
	for (int i = lane; i < nData; i += WARPSIZE) {

		distance = runDistanceFunction(func, &dataPoints[i*dimensions], queryPoint, dimensions, magnitude_query);

		Point currentPoint = createPoint(i, distance);
		//for (int j = candidateSetSize-1; j >= 0 ; j--) { // simple sorting.
		//	if (currentPoint.distance < threadQueue[j].distance) {
		//		swapPoint = threadQueue[j];
		//		threadQueue[j] = currentPoint;
		//		currentPoint = swapPoint;
		//	}
		//}
		//
		////Verify that head of thread queue is not smaller than biggest k distance.
		//if (__ballot_sync(FULL_MASK,threadQueue[0].distance < maxKDistance) && i < (nData - 1) - WARPSIZE) {
		//	startSort(threadQueue, swapPoint, params);
		//	maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance); 
		//	//printQueue(threadQueue);
		//}

		//With buffer 
		if (currentPoint.distance < maxKDistance) {
			threadQueue[queuePosition++] = currentPoint;
		}



		if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
			//printQueue(threadQueue);
			queuePosition = 0;
		}


	}

	startSort(threadQueue, swapPoint, params);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;

	for (int i = kIdx; i < k; i += WARPSIZE)
	{
		result[resultIdx + i] = threadQueue[warpQueueIdx--];
	}

}
