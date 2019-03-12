#include "point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "point.h"
#include<iostream>
#include "pointExtensions.cuh"
#include <time.h>

#define THREADS 320
#define THREAD_QUEUE_SIZE 16
#define WARPSIZE 32
#define FULL_MASK 0xffffffff
#define WARP_LEADER_THREAD 0

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

__inline__ __device__
float broadCastMaxK(float maxK) {
	float otherVal = __shfl_sync(FULL_MASK, maxK, WARP_LEADER_THREAD);
	return maxK < otherVal ? otherVal : maxK; 
}

__inline__ __device__
void subSort(Point& val, int size, Point swapPoint) {

	for (int offset = size / 2; offset > 0; offset /= 2) {
		//int otherTid = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, size);
		swapPoint.ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, size);
		swapPoint.distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, size);
		val = threadIdx.x < __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, size) ? max(val, swapPoint) : min(val, swapPoint);
	}
}

__inline__ __device__
void subSortUnrolled(Point& val, Point swapPoint) {

#pragma unroll
	for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
		//int otherTid = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, WARPSIZE);
		swapPoint.ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, WARPSIZE);
		swapPoint.distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, WARPSIZE);
		val = threadIdx.x < __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, WARPSIZE) ? max(val, swapPoint) : min(val, swapPoint);
	}
}

//__inline__ __device__
//void printThreadQueue(Point* val) {
//	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
//		printf("T[%d] TQ[%d] = (%d, %f)\n", threadIdx.x, i, val[i].ID, val[i].distance);
//	}
//}

__inline__ __device__
void laneStrideSort(Point* val, Point swapPoint) {
	int lane = threadIdx.x % WARPSIZE;
	int allElemSize = (THREAD_QUEUE_SIZE * WARPSIZE);
	int allIdx = 0; 
	int pairIdx = 0;
	int pairLane = 0;
	int exchangePairIdx = 0; 
	int exchangeLane = 0;
	//int pairCoupleSize = 0; 
	int elemsToExchange = 0;
	int start = 0; 
	int increment = 0; 
	int end = 0;

#pragma unroll
	for (int pairSize = 1; pairSize <= WARPSIZE / 2; pairSize *= 2) {

		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			allIdx = lane + WARPSIZE * i;
			pairIdx = allIdx / pairSize;
			pairLane = allIdx % pairSize;
			exchangePairIdx = pairIdx % 2 == 0 ? pairIdx + 1 : pairIdx - 1;
			exchangeLane = (exchangePairIdx * pairSize + (pairSize - pairLane - 1)) % WARPSIZE;
			swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, exchangeLane, WARPSIZE);
			swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, exchangeLane, WARPSIZE);

			//printf("(I,P): (%d,%d)  -  T[%d] to T[%d] vals: (%d,%d) \n ", i, pairSize, tid, exchangeLane, val[i], otherVal);
			val[i] = lane < exchangeLane ? max(val[i], swapPoint) : min(val[i], swapPoint);

			subSort(val[i], pairSize * 2, createPoint(-1, 10));
		}
	}


#pragma unroll
	for (int pairSize = WARPSIZE; pairSize <= (THREAD_QUEUE_SIZE * WARPSIZE) / 2; pairSize *= 2) {

		exchangeLane = (WARPSIZE - 1) - lane;
		//pairCoupleSize = (allElemSize / pairSize) / 2;
		elemsToExchange = pairSize / WARPSIZE * 2;

		for (int pairCouple = 0; pairCouple < ((allElemSize / pairSize) / 2); pairCouple++) {

			start = lane % 2 == 0 ? pairCouple * elemsToExchange : pairCouple * elemsToExchange + elemsToExchange - 1;
			increment = lane % 2 == 0 ? 1 : -1;
			end = elemsToExchange + (pairCouple * elemsToExchange);
			for (int i = start; i < end && i >= pairCouple * elemsToExchange; i += increment) {
				allIdx = lane + WARPSIZE * i;
				pairIdx = allIdx / pairSize;
				swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, exchangeLane, WARPSIZE);
				swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, exchangeLane, WARPSIZE);
				val[i] = pairIdx % 2 == 0 ? max(val[i], swapPoint) : min(val[i], swapPoint);
			}
			if (pairSize > WARPSIZE) {
				for (int i = pairCouple * elemsToExchange; i < pairCouple*elemsToExchange + elemsToExchange; i++) {
					for (int j = i; j < pairCouple*elemsToExchange + elemsToExchange; j++) {
						if (val[i].distance < val[j].distance) {
							swapPoint = val[i];
							val[i] = val[j];
							val[j] = swapPoint;
						}
					}
				}
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			subSortUnrolled(val[i], swapPoint);
		}
	}

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
	Point swapPoint;
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
		if (__ballot_sync(FULL_MASK,threadQueue[0].distance < maxKDistance)) {
			for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
				for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
					if (threadQueue[i].distance < threadQueue[j].distance) {
						swapPoint = threadQueue[j];
						threadQueue[j] = threadQueue[i];
						threadQueue[i] = swapPoint;
					}
				}
			}
			laneStrideSort(threadQueue, swapPoint);

			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance); 
		}
		
	}

	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
			if (threadQueue[i].distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = threadQueue[i];
				threadQueue[i] = swapPoint;
			}
		}
	}
	

	laneStrideSort(threadQueue, swapPoint);

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
	clock_t before = clock();
	knn << <blocks, THREADS >> > (dev_query_points, dev_data_points, N_query, N_data, d, k, dev_result);
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