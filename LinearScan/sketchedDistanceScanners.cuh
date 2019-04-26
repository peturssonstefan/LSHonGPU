#pragma once
#include "sortingFramework.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "constants.cuh"
#include "pointExtensions.cuh"
#include "candidateSetScanner.cuh"
#include "launchDTO.h"
#include "distanceFunctions.cuh"

__inline__ __device__
void scanHammingDistance(float* originalData, float* originalQuery, int dimensions, unsigned int* data, unsigned int* queries, int sketchDim, int nData, int N_query, int k, int distFunc, int implementation, Point* result)
{
	
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE; 
	Parameters params; 
	params.lane = lane; 
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;
	float maxKDistance = (float)INT_MAX;
	int warpQueueSize = k / WARPSIZE;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;
	int queuePosition = 0;

//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	for (int i = lane; i < nData; i += WARPSIZE) {
		int hammingDistance = 0;

		hammingDistance = runSketchedDistanceFunction(implementation, &data[sketchDim*i], &queries[queryIdx], sketchDim);

		Point currentPoint = createPoint(i, (float)hammingDistance); 

		if (WITH_TQ_OR_BUFFER) {
			//run TQ
			for (int j = candidateSetSize - 1; j >= 0; j--) { // simple sorting.
				if (currentPoint.distance < threadQueue[j].distance) {
					swapPoint = threadQueue[j];
					threadQueue[j] = currentPoint;
					currentPoint = swapPoint;
				}
			}

			//Verify that head of thread queue is not smaller than biggest k distance.
			if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
			}
		}
		else {
			//run buffer
			if (currentPoint.distance < maxKDistance || same(currentPoint, maxKDistance)) {
				if (queuePosition >= THREAD_QUEUE_SIZE) printf("Value larger than queue pos \n"); 
				threadQueue[queuePosition++] = currentPoint;
			}

			if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
				//printQueue(threadQueue);
				queuePosition = 0;
			}
		}
	}

	
	//Sort before candidateSetScan if we only do exact calculations on warp queue elements.

	//Candidate set scan.
	candidateSetScan(originalData, originalQuery, dimensions, threadQueue, k, distFunc);

	startSort(threadQueue, swapPoint, params);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;


	for (int i = kIdx; i < k; i += WARPSIZE)
	{
		result[resultIdx + i] = threadQueue[warpQueueIdx--];
	}
}

__inline__ __device__
void scanHammingDistanceJL(LaunchDTO<float> launchDTO)
{
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE;
	Parameters params;
	params.lane = lane;
	int queuePosition = 0; 
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * launchDTO.k;
	int queryIdx = warpId * launchDTO.sketchDim;
	int queryIdxOriginal = warpId * launchDTO.dimensions; 
	float maxKDistance = (float)INT_MAX;
	int warpQueueSize = launchDTO.k / WARPSIZE;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;


	//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	for (int i = lane; i < launchDTO.N_data; i += WARPSIZE) {
		float distance = 0;
		
		distance = runSketchedDistanceFunction(launchDTO.implementation, &launchDTO.sketchedData[launchDTO.sketchDim*i], &launchDTO.sketchedQueries[queryIdx], launchDTO.sketchDim); 


		Point currentPoint = createPoint(i, distance);

		if (WITH_TQ_OR_BUFFER) {
			//run TQ
			for (int j = candidateSetSize - 1; j >= 0; j--) { // simple sorting.
				if (currentPoint.distance < threadQueue[j].distance) {
					swapPoint = threadQueue[j];
					threadQueue[j] = currentPoint;
					currentPoint = swapPoint;
				}
			}

			//Verify that head of thread queue is not smaller than biggest k distance.
			if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
			}
		}
		else {
			//run buffer
			if (currentPoint.distance < maxKDistance || same(currentPoint, maxKDistance)) {
				threadQueue[queuePosition++] = currentPoint;
			}

			if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
				//printQueue(threadQueue);
				queuePosition = 0;
			}
		}
	}


	//Sort before candidateSetScan if we only do exact calculations on warp queue elements.

	//Candidate set scan.
	candidateSetScan(launchDTO.data, &launchDTO.queries[queryIdxOriginal], launchDTO.dimensions, threadQueue, launchDTO.k, 1);

	startSort(threadQueue, swapPoint, params);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;

	for (int i = kIdx; i < launchDTO.k; i += WARPSIZE)
	{
		launchDTO.results[resultIdx + i] = threadQueue[warpQueueIdx--];
	}
}


template<class T> __inline__ __device__
float jaccardSim(T data, T query) {
	return data == query ? 1 : 0;
}

template<class T> __inline__ __device__
float jaccardSimOneBit(T data, T query) {
	return SKETCH_COMP_SIZE - __popc(data ^ query);
}


template<class T> __inline__ __device__
void scanJaccardDistance(float* originalData, float* originalQuery, int dimensions, T * data, T * queries, int sketchDim, int nData, int N_query, int k, int distFunc, int implementation,Point* result)
{
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE;
	Parameters params;
	params.lane = lane;
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;
	float maxKDistance = (float)INT_MAX;
	int warpQueueSize = k / WARPSIZE;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	bool sketchTypeOneBit = sizeof(T) > 1; 
	int similarityDivisor = sketchTypeOneBit ? sketchDim * SKETCH_COMP_SIZE : sketchDim; 
	Point swapPoint;
	int queuePosition = 0;


	//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	for (int i = lane; i < nData; i += WARPSIZE) {

		float jaccardDistance = runSketchedDistanceFunction(implementation, &data[sketchDim*i], &queries[queryIdx], sketchDim, similarityDivisor);

		Point currentPoint = createPoint(i, jaccardDistance);

		if (WITH_TQ_OR_BUFFER) {
			//run TQ
			for (int j = candidateSetSize - 1; j >= 0; j--) { // simple sorting.
				if (currentPoint.distance < threadQueue[j].distance) {
					swapPoint = threadQueue[j];
					threadQueue[j] = currentPoint;
					currentPoint = swapPoint;
				}
			}

			//Verify that head of thread queue is not smaller than biggest k distance.
			if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
			}
		}
		else {
			//run buffer
			if (currentPoint.distance < maxKDistance || same(currentPoint, maxKDistance)) {
				threadQueue[queuePosition++] = currentPoint;
			}

			if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, params);
				maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
				//printQueue(threadQueue);
				queuePosition = 0;
			}
		}

	}


	//Sort before candidateSetScan if we only do exact calculations on warp queue elements.

	//Candidate set scan.
	candidateSetScan(originalData, originalQuery, dimensions, threadQueue, k, distFunc);

	startSort(threadQueue, swapPoint, params);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;

	for (int i = kIdx; i < k; i += WARPSIZE)
	{
		result[resultIdx + i] = threadQueue[warpQueueIdx--];
	}

}