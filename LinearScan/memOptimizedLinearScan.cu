#include "point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "point.h"
#include<iostream>
#include "pointExtensions.cuh"

#define THREADS 32
#define THREAD_QUEUE_SIZE 4
#define K 32
#define WARPSIZE 32
#define FULL_MASK 0xffffffff

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\


__inline__ __device__
Point* divideData(Point* val) {
	static __shared__ Point allData[THREAD_QUEUE_SIZE * WARPSIZE]; //nederen

	// write to shared memory
	int valIdx = 0;
	for (int i = threadIdx.x * THREAD_QUEUE_SIZE; i < threadIdx.x * THREAD_QUEUE_SIZE + THREAD_QUEUE_SIZE; i++) {
		allData[i] = val[valIdx++];
	}

	__syncthreads();

	valIdx = 0;
	// read to local memory
	for (int i = threadIdx.x; i < THREAD_QUEUE_SIZE * WARPSIZE; i += WARPSIZE) {
		val[valIdx++] = allData[i];
	}

	return val;
}



__inline__ __device__
Point* warpSort(Point* val) {

	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		int otherThreadId = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset);

		int start = threadIdx.x < otherThreadId ? THREAD_QUEUE_SIZE - 1 : 0;
		int increment = threadIdx.x < otherThreadId ? -1 : 1;

		for (int i = start; i < THREAD_QUEUE_SIZE && i >= 0; i += increment) {
			int othervalID = __shfl_xor_sync(FULL_MASK, val[i].ID, offset);
			float othervalDistance = __shfl_xor_sync(FULL_MASK, val[i].distance, offset);
			Point otherPoint = createPoint(othervalID, othervalDistance);
			val[i] = threadIdx.x < otherThreadId ? max(val[i], otherPoint) : min(val[i], otherPoint);
		}

		int startIndex = 0;
		int stride = THREAD_QUEUE_SIZE / 2;
		while (true) {
			if (stride == 0) break;
			for (int i = startIndex; i < startIndex + stride; i++) {
				Point v1 = val[i];
				Point v2 = val[i + stride];
				if (v1.distance < v2.distance) {
					val[i + stride] = v1;
					val[i] = v2;
				}
			}
			if (startIndex + 2 * stride >= THREAD_QUEUE_SIZE) {
				startIndex = 0;
				stride /= 2;
			}
			else {
				startIndex += stride * 2;
			}
		}
	}

	val = divideData(val);

	return val;
}



__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* result) {
	Point threadQueue[THREAD_QUEUE_SIZE]; 
	int threadId = threadIdx.x; 
	int queryId = (threadIdx.x / warpSize) * dimensions;
	int lane = threadIdx.x % warpSize; 


	//Fill thread queue with defaults
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		Point p;
		p.distance = 2.0f;
		p.ID = -1;
	}

	float magnitude_query = 0;

	for (int j = 0; j < dimensions; j++) {
		magnitude_query += queryPoints[queryId + j] * queryPoints[queryId + j];
	}

	magnitude_query = sqrt(magnitude_query);

	//Iterate over data; 
	for (int i = lane; i < nData; i += warpSize) {
		float dotProduct = 0;
		float magnitude_data = 0.0;

		for (int j = 0; j < dimensions; j++) {
			dotProduct += queryPoints[queryId + j] * dataPoints[dimensions*i + j];
			magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
		}

		magnitude_data = sqrt(magnitude_data);
		float angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		Point currentPoint; 
		currentPoint.distance = angular_distance; 
		currentPoint.ID = i; 

		Point swapPoint;
		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (threadQueue[j].distance > currentPoint.distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

	}



}

Point* runMemOptimizedLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries) {
	CUDA_CHECK_RETURN(cudaSetDevice(0));
	int threads = 32;
	int blocks = 1;
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

	knn << <blocks, threads>> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_result);

	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}