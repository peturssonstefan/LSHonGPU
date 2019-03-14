#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<iostream>
#include <stdio.h>
#include <queue>
#include "point.h"
#include <time.h>
#include <algorithm>
#include "sortingFramework.cuh"



#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* threadQueue, Point* result) {
	Point candidateItems[MAX_K]; 
	
#pragma unroll
	for (int i = 0; i < MAX_K; i++) {
		Point p; 
		p.distance = 2.0f; 
		p.ID = -1;
		candidateItems[i] = p;
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x; 
	int queryIndex = blockIdx.x * dimensions;
	int tIndex = threadId * k;

	float magnitude_query = 0.0;

#pragma unroll
	for (int j = 0; j < dimensions; j++) {
		magnitude_query += queryPoints[queryIndex + j] * queryPoints[queryIndex + j];
	}

	magnitude_query = sqrt(magnitude_query);

	for (int i = threadIdx.x; i < nData; i += blockDim.x) {
		float dotProduct = 0;
		float magnitude_data = 0.0;

#pragma unroll
		for (int j = 0; j < dimensions; j++) {
			dotProduct += queryPoints[queryIndex + j] * dataPoints[dimensions*i + j];
			magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
		}
		
		magnitude_data = sqrt(magnitude_data);
		float angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		Point currentPoint;

		currentPoint.ID = i;
		currentPoint.distance = angular_distance;

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

	Point* kNearest = blockReduce(&threadQueue[tIndex], k, 2.0f);

	if (threadIdx.x == 0) {

#pragma unroll
		for (int i = 0; i < k; i++) {
			result[blockIdx.x * k + i] = kNearest[i];
		}
	}

}


Point* runOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries) {

	CUDA_CHECK_RETURN(cudaSetDevice(0));
	int threads = 1024;
	int blocks = 10000; //TODO, set blocks equal to query points.

	// Set up result array
	int resultSize = blocks * k; 
	Point *resultArray = (Point*)malloc(resultSize * sizeof(Point));

	// Set up thread queue array
	int threadQueueSize = threads * blocks * k;
	Point* threadQueueArray = (Point*)malloc(threadQueueSize * sizeof(Point));
	
	
	// Set up cuda device arrays

	// queries
	float* dev_query_points = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_query_points, N_query * d * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_query_points, queries, N_query * d * sizeof(float), cudaMemcpyHostToDevice));

	// data
	float* dev_data_points = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_data_points, N_data * d * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_data_points, data, N_data * d * sizeof(float), cudaMemcpyHostToDevice));

	// thread queue
	Point* dev_thread_queue;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_thread_queue, threadQueueSize * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_thread_queue, threadQueueArray, threadQueueSize * sizeof(Point), cudaMemcpyHostToDevice));

	// result
	Point* dev_result = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_result, resultSize * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_result, resultArray, resultSize * sizeof(Point), cudaMemcpyHostToDevice));

	clock_t before = clock();
	knn << <blocks, threads >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_thread_queue ,dev_result);

	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	clock_t time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	CUDA_CHECK_RETURN(cudaMemcpy(resultArray, dev_result, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

	//Free memory... 
	CUDA_CHECK_RETURN(cudaFree(dev_query_points));
	CUDA_CHECK_RETURN(cudaFree(dev_data_points));
	CUDA_CHECK_RETURN(cudaFree(dev_thread_queue));
	CUDA_CHECK_RETURN(cudaFree(dev_result));

	free(threadQueueArray);

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return resultArray;
}