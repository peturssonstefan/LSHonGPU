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


#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

__inline__ __host__ __device__
void printQueue(Point* queue) {
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		printf("T[%d] arr[%d] = (%d,%f) \n", threadIdx.x, i, queue[i].ID, queue[i].distance);
	}
}

__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* result) {
	
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
	//Fill thread queue with defaults
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	float magnitude_query = 0;
	float dotProduct = 0;
	float magnitude_data = 0.0;
	float angular_distance = 0.0;

#pragma unroll
	for (int j = 0; j < dimensions; j++) {
		magnitude_query += queryPoints[queryId + j] * queryPoints[queryId + j];
	}

	magnitude_query = sqrt(magnitude_query);

	//Iterate over data; 
	for (int i = lane; i < nData; i += WARPSIZE) {
		dotProduct = 0; // reset value.
		magnitude_data = 0.0; // reset value.
		angular_distance = 0.0; // reset value.

#pragma unroll
		for (int j = 0; j < dimensions; j++) {
			dotProduct += queryPoints[queryId + j] * dataPoints[dimensions*i + j];
			magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
		}

		magnitude_data = sqrt(magnitude_data);
		angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		Point currentPoint = createPoint(i, angular_distance);
		for (int j = candidateSetSize-1; j >= 0 ; j--) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}
		
		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK,threadQueue[0].distance < maxKDistance && i < (nData - 1) - WARPSIZE)) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance); 
			printQueue(threadQueue);
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

Point* runMemOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries) {
	CUDA_CHECK_RETURN(cudaSetDevice(0));
	int numberOfThreads = 32;//calculateThreadsLocal(N_query);
	int numberOfBlocks = calculateBlocksLocal(N_query);
	int resultSize = N_query * k;
	Point *resultArray = (Point*)malloc(resultSize * sizeof(Point));
	// queries
	float* dev_query_points = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_query_points, N_query * d * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_query_points, queries, N_query * d * sizeof(float), cudaMemcpyHostToDevice));

	// data
	float* dev_data_points = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_data_points, N_data * d * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_data_points, data, N_data * d * sizeof(float), cudaMemcpyHostToDevice));

	// result
	Point* dev_result = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_result, resultSize * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_result, resultArray, resultSize * sizeof(Point), cudaMemcpyHostToDevice));
	clock_t before = clock();
	knn << <numberOfBlocks, numberOfThreads >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_result);
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	clock_t time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	CUDA_CHECK_RETURN(cudaMemcpy(resultArray, dev_result, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

	//Free memory... 
	CUDA_CHECK_RETURN(cudaFree(dev_query_points));
	CUDA_CHECK_RETURN(cudaFree(dev_data_points));
	CUDA_CHECK_RETURN(cudaFree(dev_result));

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return resultArray; 
}