#include "point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "point.h"
#include<iostream>
#include "pointExtensions.cuh"
#include <time.h>
#include <math.h>
#include "constants.cuh"
#include "sortParameters.h"
#include "sortingFramework.cuh"
#include "launchHelper.cuh"
#include "processingUtils.cuh"
#include "distanceFunctions.cuh"
#include "cudaHelpers.cuh"

__inline__ __host__ __device__
void printQueue(Point* queue) {
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		printf("T[%d] arr[%d] = (%d,%f) \n", threadIdx.x, i, queue[i].ID, queue[i].distance);
	}
}

__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* result, int func) {
	
	Point threadQueue[THREAD_QUEUE_SIZE];
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryId = warpId * dimensions;
	if (queryId > nQueries * dimensions) return; 
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
		magnitude_query += queryPoints[queryId + j] * queryPoints[queryId + j];
	}

	magnitude_query = sqrt(magnitude_query);

	//Iterate over data; 
	for (int i = lane; i < nData; i += WARPSIZE) {

		distance = runDistanceFunction(func, &dataPoints[i*dimensions], &queryPoints[queryId], dimensions, magnitude_query);

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

		

		if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK ) {
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

__global__
void normalizeData(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions) {
	transformToUnitVectors(queryPoints, nQueries, dimensions);
	transformToUnitVectors(dataPoints, nData, dimensions);
}

__global__
void preprocess(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int* minValues)
{
	transformData(dataPoints, queryPoints, nData, nQueries, dimensions, minValues);
}

Point* runMemOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries, int distanceFunc) {
	setDevice();
	int numberOfThreads = calculateThreadsLocal(N_query);
	int numberOfBlocks = calculateBlocksLocal(N_query);
	int resultSize = N_query * k;
	Point *resultArray = (Point*)malloc(resultSize * sizeof(Point));
	
	// queries
	float* dev_query_points = mallocArray(queries, N_query * d, true);
	// data
	float* dev_data_points = mallocArray(data, N_data * d, true);

	// result
	Point* dev_result = mallocArray(resultArray, resultSize);

	if (distanceFunc == 2) {
		printf("Starting preprocess \n");
		int* dev_minValues = mallocArray<int>(nullptr, d);
		preprocess << <1, numberOfThreads >> > (dev_query_points, dev_data_points, N_query, N_data, d, dev_minValues);
		waitForKernel();

		normalizeData << < numberOfBlocks, numberOfThreads >> > (dev_query_points, dev_data_points, N_query, N_data, d);
		waitForKernel();
		
		printf("Done preprocessing \n");
	}

	printf("Launching KNN \n");
	clock_t before = clock();
	knn << <numberOfBlocks, numberOfThreads >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_result, distanceFunc);
	waitForKernel();

	clock_t time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	
	copyArrayToHost(resultArray, dev_result, resultSize);

	//for (int i = 0; i < N_data; i++) {
	//	printf("%d ", i);
	//	for (int j = 0; j < d; j++) {
	//		printf("%f ", data[i * d + j]);
	//	}
	//	printf("\n");
	//}

	//Free memory... 
	freeDeviceArray(dev_query_points);
	freeDeviceArray(dev_data_points);
	freeDeviceArray(dev_result);

	resetDevice();

	return resultArray; 
}