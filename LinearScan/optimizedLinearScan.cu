#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<iostream>
#include "gloveparser.cuh"
#include <stdio.h>
#include "point.h"
#include <time.h>
#include <algorithm>

#define FULL_MASK 0xffffffff

template <typename T>
inline __device__ T* shuffle_down(T* const val, unsigned int delta, int width = warpSize) {
	//static assert(sizeof(T*) == sizeof(long long), "pointer size incorrect"); 
	long long v = (long long)val;
	return (T*)__shfl_down_sync(FULL_MASK, v, delta);
}

__inline__ __device__
Point* warpReduceArrays(Point* val, int k) {
	Point merged[100];
	for (int offset = 16; offset > 0; offset /= 2) {
		Point* tmpVal = shuffle_down(val, offset);
		int i = 0;
		int j = 0;
		//printf("Val: %d \n", tmpVal[j]); 
		for (int x = 0; x < k; x++) {
			if (val[i].distance <= tmpVal[j].distance) {
				merged[x] = val[i++];
			}
			else if (val[i].distance > tmpVal[j].distance) {
				merged[x] = tmpVal[j++];
			}
		}

		for (int i = 0; i < k; i++) {
			val[i] = merged[i];
		}
	}

	return val;
}

__inline__ __device__
Point* blockReduce(Point* val, int k) {
	__shared__ Point* shared[32];

	int lane = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;

	val = warpReduceArrays(val, k);

	if (lane == 0) {
		shared[warpId] = val;
	}

	__syncthreads();

	static __shared__ Point maxArray[10];

	for (int i = 0; i < 10; i++) {
		Point p;
		p.ID = -1;
		p.distance = 2.0f;
		maxArray[i] = p;
	}


	val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : maxArray;


	if (warpId == 0) {
		val = warpReduceArrays(val, k);
	}

	return val;
}

__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* threadQueue, Point* result) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x; 
	int queryIndex = blockIdx.x * dimensions;
	int tIndex = threadId * k;
	for (int i = threadIdx.x; i < nData; i += blockDim.x) {
		float dotProduct = 0;
		float magnitude_query = 0.0;
		float magnitude_data = 0.0;
		for (int j = 0; j < dimensions; j++) {
			dotProduct += queryPoints[queryIndex + j] * dataPoints[dimensions*i + j];
			magnitude_query += queryPoints[queryIndex + j] * queryPoints[queryIndex + j];
			magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
		}

		magnitude_query = sqrt(magnitude_query);
		magnitude_data = sqrt(magnitude_data);
		float angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		Point currentPoint;

		currentPoint.ID = i;
		currentPoint.distance = angular_distance;

		Point swapPoint;
		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (threadQueue[tIndex + j].distance > currentPoint.distance) {
				swapPoint = threadQueue[tIndex + j];
				threadQueue[tIndex + j] = currentPoint;
				currentPoint = swapPoint;
			}
		}
	}

	Point* kNearest = blockReduce(&threadQueue[tIndex], k);

	if (threadIdx.x == 0) {
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
	int blocks = 10000;

	// Set up result array
	int resultSize = blocks * k; 
	Point *resultArray = (Point*)malloc(resultSize * sizeof(Point));

	// Set up thread queue array
	int threadQueueSize = threads * blocks * k;
	Point* threadQueueArray = (Point*)malloc(threadQueueSize * sizeof(Point));

	// Set default data into thread queues
	for (int i = 0; i < threadQueueSize; i++) {
		Point p;
		p.ID = -1;
		p.distance = 2.0f; //fill thread queue array with default max value - given sim [-1,1]

		threadQueueArray[i] = p;
	}

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

	/*for (int queryId = 0; queryId < blocks; queryId++) {
		printf("query: %d \n", queryId);
		for (int i = 0; i < k; i++) {
			printf("Id: %d - %f\n", resultArray[queryId * k + i].ID, resultArray[queryId * k + i].distance);
		}
	}*/

	printf("Done. \n");

	//Free memory... 
	cudaFree(dev_query_points);
	cudaFree(dev_data_points);
	cudaFree(dev_thread_queue);
	cudaFree(dev_result);

	free(threadQueueArray);

	cudaStatus = cudaDeviceReset();

	return resultArray;
}