#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<iostream>
#include <stdio.h>
#include <queue>
#include "point.h"
#include <time.h>
#include <algorithm>
#include "shuffleUtils.cuh"


__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* threadQueue, Point* result) {
	Point candidateItems[MAX_K]; 
	
#pragma unroll
	for (int i = 0; i < MAX_K; i++) {
		Point p; 
		p.distance = 2.0f; 
		p.ID = -1;
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

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Is there a CUDA-capable GPU installed?");
		throw "Error in simpleLinearScan run.";
	}
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
	cudaMalloc((void**)&dev_query_points, N_query * d * sizeof(float));
	cudaMemcpy(dev_query_points, queries, N_query * d * sizeof(float), cudaMemcpyHostToDevice);

	// data
	float* dev_data_points = 0;
	cudaMalloc((void**)&dev_data_points, N_data * d * sizeof(float));
	cudaMemcpy(dev_data_points, data, N_data * d * sizeof(float), cudaMemcpyHostToDevice);

	// thread queue
	Point* dev_thread_queue;
	cudaMalloc((void**)&dev_thread_queue, threadQueueSize * sizeof(Point));
	cudaMemcpy(dev_thread_queue, threadQueueArray, threadQueueSize * sizeof(Point), cudaMemcpyHostToDevice);

	// result
	Point* dev_result = 0;
	cudaMalloc((void**)&dev_result, resultSize * sizeof(Point));
	cudaMemcpy(dev_result, resultArray, resultSize * sizeof(Point), cudaMemcpyHostToDevice);

	printf("Threads: %d\n", threads);
	printf("Blocks: %d\n", blocks);
	clock_t before = clock();
	knn << <blocks, threads >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_thread_queue ,dev_result);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		throw "Error in optimizedLinearScan run.";
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		throw "Error in optimizedLinearScan run.";
	}

	clock_t time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	cudaStatus = cudaMemcpy(resultArray, dev_result, resultSize * sizeof(Point), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy from device to host returned error code %d \n", cudaStatus);
		throw "Error in optimizedLinearScan run.";
	}

	//Free memory... 
	cudaFree(dev_query_points);
	cudaFree(dev_data_points);
	cudaFree(dev_thread_queue);
	cudaFree(dev_result);

	free(threadQueueArray);

	cudaStatus = cudaDeviceReset();

	return resultArray;
}