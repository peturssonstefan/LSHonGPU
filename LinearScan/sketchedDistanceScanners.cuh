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

__inline__ __device__
void scanHammingDistance(float* originalData, float* originalQuery, int dimensions, unsigned int* data, unsigned int* queries, int sketchDim, int nData, int N_query, int k, int distFunc,Point* result)
{
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE; 
	Parameters params; 
	params.lane = lane; 
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;
	int maxKDistance = 0;
	int warpQueueSize = k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;


//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, SKETCH_COMP_SIZE * sketchDim + 1);
	}

	for (int i = lane; i < nData; i += WARPSIZE) {
		int hammingDistance = 0;

//#pragma unroll
		for (int j = 0; j < sketchDim; j++) {
			unsigned int queryValue = queries[queryIdx + j];
			unsigned int dataValue = data[sketchDim*i + j]; 
			unsigned int bits = queryValue ^ dataValue; 
			int bitCount = __popc(bits);
			hammingDistance += bitCount;
		}

		Point currentPoint = createPoint(i, (float)hammingDistance); 

		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
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
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * launchDTO.k;
	int queryIdx = warpId * launchDTO.sketchDim;
	int queryIdxOriginal = warpId * launchDTO.dimensions; 
	int maxKDistance = 0;
	int warpQueueSize = launchDTO.k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	Point swapPoint;


	//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, (float)INT_MAX);
	}

	for (int i = lane; i < launchDTO.N_data; i += WARPSIZE) {
		float dotProduct = 0;

		//#pragma unroll
		for (int j = 0; j < launchDTO.sketchDim; j++) {
			
			float queryVal = launchDTO.sketchedQueries[queryIdx + j];
			float dataVal = launchDTO.sketchedData[launchDTO.sketchDim*i + j];
			//float dist = queryVal * dataVal;
			float dist = pow((queryVal - dataVal),2); 
			//float dist = abs(queryVal - dataVal);
			dotProduct += dist;
		}
		

		Point currentPoint = createPoint(i, dotProduct);

		for (int j = 0; (j < launchDTO.k && j <= i); j++) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
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
void scanJaccardDistance(float* originalData, float* originalQuery, int dimensions, T * data, T * queries, int sketchDim, int nData, int N_query, int k, int distFunc,Point* result)
{
	Point threadQueue[THREAD_QUEUE_SIZE];
	int lane = threadIdx.x % WARPSIZE;
	Parameters params;
	params.lane = lane;
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int resultIdx = warpId * k;
	int queryIdx = warpId * sketchDim;
	int maxKDistance = 0;
	int warpQueueSize = k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	bool sketchTypeOneBit = sizeof(T) > 1; 
	int similarityDivisor = sketchTypeOneBit ? sketchDim * SKETCH_COMP_SIZE : sketchDim; 
	Point swapPoint;


	//#pragma unroll
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, SKETCH_COMP_SIZE * sketchDim + 1);
	}

	for (int i = lane; i < nData; i += WARPSIZE) {
		float jaccardSimilarity = 0;

		for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {
			T dataSketch = data[sketchDim*i + hashIdx];
			T querySketch = queries[queryIdx + hashIdx];
			jaccardSimilarity += sketchTypeOneBit ? jaccardSimOneBit(dataSketch, querySketch) : jaccardSim(dataSketch, querySketch);
		}


		jaccardSimilarity /= similarityDivisor;

		float jaccardDistance = 1 - jaccardSimilarity;

		//printf("Jaccard distance: %f \n", jaccardDistance);

		Point currentPoint = createPoint(i, jaccardDistance);

		for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}

		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
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