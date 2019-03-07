#include "point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "point.h"
#include<iostream>
#include "pointExtensions.cuh"

#define THREADS 320
#define THREAD_QUEUE_SIZE 4
#define K 32
#define WARPSIZE 32
#define FULL_MASK 0xffffffff
#define WARP_LEADER_THREAD 0

#define DEBUG_WARP 1

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\


__inline__ __device__
Point* divideData(Point* val) {
	int lane = threadIdx.x % WARPSIZE;
	static __shared__ Point allData[THREAD_QUEUE_SIZE * WARPSIZE*(THREADS / WARPSIZE)]; //nederen

	int warpArrSize = THREAD_QUEUE_SIZE * WARPSIZE;

	int warpID = threadIdx.x / WARPSIZE;

	int warpEndIdx = warpID * warpArrSize + warpArrSize;

	// write to shared memory
	int valIdx = 0;
	for (int i = threadIdx.x * THREAD_QUEUE_SIZE; i < threadIdx.x * THREAD_QUEUE_SIZE + THREAD_QUEUE_SIZE; i++) {
		allData[i] = val[valIdx++];
	}

	__syncwarp();

	valIdx = 0;
	// read to local memory
	for (int i = (warpID * warpArrSize) + lane; i < warpEndIdx; i += WARPSIZE) {
		val[valIdx++] = allData[i];
	}

	return val;
}

__inline__ __device__
float broadCastMaxK(float maxK) {
	float otherVal = __shfl_sync(FULL_MASK, maxK, WARP_LEADER_THREAD);
	return maxK < otherVal ? otherVal : maxK; 
}


__inline__ __device__
Point subPairSort(Point val, int wSize) {
	int lane = threadIdx.x % wSize;

	for (int offset = wSize / 2; offset > 0; offset /= 2) {
		//printf("Offset: %d \n", offset);
		Point otherPoint;

		int otherT = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset);
		int otherLane = __shfl_xor_sync(FULL_MASK, lane, offset);
		otherPoint.ID = __shfl_xor_sync(FULL_MASK, val.ID, offset);
		otherPoint.distance = __shfl_xor_sync(FULL_MASK, val.distance, offset);

		val = lane < otherLane ? max(val, otherPoint) : min(val, otherPoint);

	}
	return val;
}

__inline__ __device__
void printThreadQueue(Point* val) {
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		printf("T[%d] TQ[%d] = (%d, %f)\n", threadIdx.x, i, val[i].ID, val[i].distance);
	}
}

__inline__ __device__
Point* warpSort(Point* val) {
	int lane = threadIdx.x % WARPSIZE;
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE; 

	/*if (warpId == 1) {
		printf("Before sort \n");
		printThreadQueue(val);
	}*/


	for (int pairSize = 1; pairSize < WARPSIZE; pairSize *= 2) {
		//printf("Pair size: %d\n", pairSize);

		int pairIdx = lane / pairSize;
		int pairLane = lane % pairSize;
		int exchangePairIdx = pairIdx % 2 == 0 ? pairIdx + 1 : pairIdx - 1;
		int exchangeLane = (exchangePairIdx * pairSize + (pairSize - pairLane - 1));
		int start = pairIdx % 2 == 0 ? 0 : THREAD_QUEUE_SIZE - 1;
		int increment = pairIdx % 2 == 0 ? 1 : -1;

		for (int i = start; i < THREAD_QUEUE_SIZE && i >= 0; i += increment) {
			Point otherPoint;
			otherPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, exchangeLane);
			otherPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, exchangeLane);
			
			val[i] = lane < exchangeLane ? max(val[i], otherPoint) : min(val[i], otherPoint);

			val[i] = subPairSort(val[i], pairSize * 2);
		}

		// Local sort.
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

	/*if (warpId == 1) {
		printf("After sort \n");
		printThreadQueue(val);
	}*/

	return val;
}

__global__
void knn(float* queryPoints, float* dataPoints, int nQueries, int nData, int dimensions, int k, Point* result) {
	Point threadQueue[THREAD_QUEUE_SIZE];
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryId = warpId * dimensions;
	int lane = threadIdx.x % WARPSIZE;
	float maxKDistance = (float)INT_MAX; 
	int warpQueueSize = k / WARPSIZE; 
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;

	printf("T[%d] queryID: %d\n", threadIdx.x, queryId);

	//Fill thread queue with defaults
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	float magnitude_query = 0;

	for (int j = 0; j < dimensions; j++) {
		magnitude_query += queryPoints[queryId + j] * queryPoints[queryId + j];
	}

	magnitude_query = sqrt(magnitude_query);

	//Iterate over data; 
	for (int i = lane; i < nData; i += WARPSIZE) {
		float dotProduct = 0;
		float magnitude_data = 0.0;

		for (int j = 0; j < dimensions; j++) {
			dotProduct += queryPoints[queryId + j] * dataPoints[dimensions*i + j];
			magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
		}

		magnitude_data = sqrt(magnitude_data);
		float angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

		Point currentPoint = createPoint(i, angular_distance);
		Point swapPoint;
		for (int j = candidateSetSize-1; j >= 0 ; j--) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}
		
		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK,threadQueue[0].distance < maxKDistance)) {
			for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
				for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
					if (threadQueue[i].distance < threadQueue[j].distance) {
						Point swap = threadQueue[j]; 
						threadQueue[j] = threadQueue[i];
						threadQueue[i] = swap; 
					}
				}
			}
			Point* newQueue = warpSort(threadQueue);
			for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
				threadQueue[i] = newQueue[i]; 
			}
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance); 
		}
		
	}

	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
			if (threadQueue[i].distance < threadQueue[j].distance) {
				Point swap = threadQueue[j];
				threadQueue[j] = threadQueue[i];
				threadQueue[i] = swap;
			}
		}
	}

	__syncthreads(); //necessary ? 
	


	Point* newQueue = warpSort(threadQueue);
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = newQueue[i];
	}

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

	knn << <blocks, THREADS >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_result);

	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(resultArray, dev_result, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

	//Free memory... 
	CUDA_CHECK_RETURN(cudaFree(dev_query_points));
	CUDA_CHECK_RETURN(cudaFree(dev_data_points));
	CUDA_CHECK_RETURN(cudaFree(dev_result));

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return resultArray; 
}