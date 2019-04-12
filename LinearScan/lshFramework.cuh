#pragma once

#include "point.h"
#include "launchDTO.h"
#include "simHash.cuh"
#include "randomVectorGenerator.h"; 
#include "device_launch_parameters.h"
#include "cudaHelpers.cuh"
#include <bitset>
#include <math.h>
#include <cuda.h>

__inline__ __host__ __device__
void printQueue(Point* queue) {
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		printf("T[%d] arr[%d] = (%d,%f) \n", threadIdx.x, i, queue[i].ID, queue[i].distance);
	}
}

template<class T>
void runSketchSimHash(LaunchDTO<T> params, short* dev_bucketKeysData, short* dev_bucketKeysQueries, int numberOfBlocks, int numberOfThreads) {
	float* randomVectors = generateRandomVectors(params.tables * params.dimensions * params.bucketKeyBits);
	float* dev_randomVectors = mallocArray(randomVectors, params.tables * params.dimensions * params.bucketKeyBits, true);
	sketchDataGeneric<<<numberOfBlocks, numberOfThreads>>>(params.data, dev_randomVectors, params.N_data, params.dimensions, params.tables, params.bucketKeyBits, dev_bucketKeysData);
	waitForKernel();
	sketchDataGeneric<<<numberOfBlocks, numberOfThreads>>>(params.queries, dev_randomVectors, params.N_queries, params.dimensions, params.tables, params.bucketKeyBits, dev_bucketKeysQueries);
	waitForKernel();
}

template <class T>
void sketchPoints(LaunchDTO<T> params, short* dev_bucketKeysData, short* dev_bucketKeysQueries, int numberOfBlocks, int numberOfThreads) {
	switch (params.implementation) {
	case 3:
		printf("Using Simhash to sketch \n");
		runSketchSimHash(params, dev_bucketKeysData, dev_bucketKeysQueries, numberOfBlocks, numberOfThreads); 
		break; 
	case 4: 
		printf("Using Weighted Minhash to sketch \n");
		break; 
	case 5:
		printf("Using 1 Bit Weighted Minhash to sketch \n");
		break; 
	case 6: 
		printf("Using Johnson Lindenstrauss to sketch \n");
		break;
	}
}

template<class T> __global__ 
void findBucketDistribution(LaunchDTO<T> params, short* dev_bucketKeysData, int bucketCount, int tableSize,int* hashKeys) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	
	for (int i = threadId; i < params.N_data; i += threadCount) {
		for (int tableIdx = 0; tableIdx < params.tables; tableIdx++) {
			int hash = dev_bucketKeysData[i*params.tables + tableIdx];
			atomicAdd(&hashKeys[tableIdx * tableSize + hash], 1);
		}
	}
}

__global__
void buildHashBucketIndexes(int N_data,int tableSize, int* hashKeys) {
	int tableIdx = threadIdx.x;
	int currentStartIdx = tableIdx * N_data;
	for (int i = 0; i < tableSize; i++) {
		int bucketSize = hashKeys[tableIdx * tableSize + i];
		hashKeys[tableIdx * tableSize + i] = currentStartIdx;
		currentStartIdx += bucketSize;
	}
}

template<class T> __global__
void distributePointsToBuckets(LaunchDTO<T> params,int tableSize, int* hashKeys, short* dev_bucketKeysData, int* bucketCounters, int* buckets) {
	int threadId = (blockIdx.x * gridDim.x) + threadIdx.x;
	int warpId = threadId / WARPSIZE;
	int lane = threadId % WARPSIZE;
	for (int i = lane; i < params.N_data; i+=WARPSIZE) {
		int hash = dev_bucketKeysData[i * params.tables + warpId];
		int bucketIdx = atomicAdd(&bucketCounters[warpId * tableSize + hash], 1);
		int bucketStart = hashKeys[warpId * tableSize + hash];

		//printf("I %d Hash: %d-> %d + %d = %d warpid %d \n", i, hash,bucketStart, bucketIdx, bucketStart + bucketIdx, warpId);

		buckets[bucketStart + bucketIdx] = i;
	}
}

template <class T> __global__ 
void scan(LaunchDTO<T> params, short* dev_bucketKeysQueries, int* dev_hashKeys, int* dev_buckets, int tableSize, Point* result) {

	Point threadQueue[THREAD_QUEUE_SIZE];

	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int warpId = threadId / WARPSIZE;
	float maxKDistance = (float)INT_MAX;
	int lane = threadId % WARPSIZE;
	int warpQueueSize = params.k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int queryHashIdx = warpId * params.tables;
	int queryIdx = warpId * params.dimensions; 
	int resultIdx = warpId * THREAD_QUEUE_SIZE * WARPSIZE;
	float magnitudeQuery = 0; 
	Point swapPoint;
	Parameters sortParameters;
	sortParameters.lane = lane; 
	int queuePosition = 0;
	
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	for (int dim = 0; dim < params.dimensions; dim++) {
		magnitudeQuery += params.queries[queryIdx + dim] * params.queries[queryIdx + dim];
	}

	magnitudeQuery = sqrt(magnitudeQuery); 
	//printf("magnitudeQuery: %f \n", magnitudeQuery); 
	

	for (int tableIdx = 0; tableIdx < params.tables; tableIdx++) {
		// Find bucket place
		int queryHashKey = dev_bucketKeysQueries[queryHashIdx + tableIdx];
		int bucketStart = dev_hashKeys[tableSize * tableIdx + queryHashKey];
		int bucketEnd = queryHashKey >= tableSize - 1 ? tableIdx * params.N_data + params.N_data : dev_hashKeys[tableIdx * tableSize + queryHashKey + 1];

		//scan bucket
		for (int bucketIdx = bucketStart + lane; bucketIdx < bucketEnd; bucketIdx += WARPSIZE) {
			int dataIdx = dev_buckets[bucketIdx];

			float distance = runDistanceFunction(params.distanceFunc, &params.data[dataIdx * params.dimensions], &params.queries[queryIdx], params.dimensions, magnitudeQuery);

			Point point = createPoint(dataIdx, distance);

			for (int i = candidateSetSize - 1; i >= 0; i--) {
				if (point.distance < threadQueue[i].distance) {
					swapPoint = threadQueue[i];
					threadQueue[i] = point;
					point = swapPoint;
				}
			}

			if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
				startSort(threadQueue, swapPoint, sortParameters);
				maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
			}

		}

		// Might need at sort here
		/*if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance)) {
			startSort(threadQueue, swapPoint, sortParameters);
			maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
		}*/
	}

	//printQueue(threadQueue);

	startSort(threadQueue, swapPoint, sortParameters);

	//Copy result from warp queues to result array in reverse order. 
	int kIdx = (WARPSIZE - lane) - 1;
	int warpQueueIdx = THREAD_QUEUE_SIZE - 1;

	for (int i = kIdx; i < THREAD_QUEUE_SIZE * WARPSIZE; i += WARPSIZE)
	{
		result[resultIdx + i] = threadQueue[warpQueueIdx--];
	}

}


template<class T> __global__
void removeDuplicates(LaunchDTO<T> params, Point* duplicateResult, Point* result) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int queryId = threadId;

	if (!(queryId < params.N_queries)) {
		return;
	}

	int resultIdx = queryId * THREAD_QUEUE_SIZE * WARPSIZE;
	int finalResulIdx = queryId * params.k;
	int endResultIdx = finalResulIdx + params.k;

	int prevId = -1; 

	for (int i = 0; i < WARPSIZE * THREAD_QUEUE_SIZE; i++) {
		if (prevId != duplicateResult[resultIdx + i].ID) {
			result[finalResulIdx++] = duplicateResult[resultIdx + i];
			prevId = duplicateResult[resultIdx + i].ID;
		}

		if (finalResulIdx >= endResultIdx) {
			break;
		}
	}
}

template<class T>
void runDistributePointsTobuckets(LaunchDTO<T> params, int tableSize, int* dev_hashKeys, short* dev_bucketKeysData, int* dev_buckets) {
	int* bucketsCounters = (int*)malloc(tableSize * params.tables * sizeof(int));

	for (int i = 0; i < tableSize * params.tables; i++) {
		bucketsCounters[i] = 0;
	}
	int* dev_bucketCounters = mallocArray(bucketsCounters, tableSize * params.tables, true);

	distributePointsToBuckets << <1, params.tables * WARPSIZE >> > (params, tableSize, dev_hashKeys, dev_bucketKeysData, dev_bucketCounters, dev_buckets);
	waitForKernel();

	freeDeviceArray(dev_bucketCounters);
	free(bucketsCounters);
}

template<class T>
Point* runLsh(LaunchDTO<T> params) {
	
	int tableSize = 1 << params.bucketKeyBits;
	int totalBucketCount = tableSize * params.tables; 
	int* hashKeys = (int*)malloc(totalBucketCount * sizeof(int));
	for (int i = 0; i < totalBucketCount; i++) {
		hashKeys[i] = 0; 
	}

	printf("tableSize: %d \n", tableSize); 

	int* dev_hashKeys = mallocArray(hashKeys, totalBucketCount, true); 
	int numberOfThreads = calculateThreadsLocal(params.N_queries);
	int numberOfBlocks = calculateBlocksLocal(params.N_queries);
	//Malloc place for bucket keys. 
	short* bucketKeysData = (short*)malloc(params.tables * params.N_data * sizeof(short)); 
	short* dev_bucketKeysData = mallocArray(bucketKeysData, params.tables * params.N_data); 
	short* bucketKeysQueries = (short*)malloc(params.tables * params.N_queries * sizeof(short));
	short* dev_bucketKeysQueries = mallocArray(bucketKeysQueries, params.tables * params.N_queries);

	printf("Building key hashes \n");
	sketchPoints(params, dev_bucketKeysData, dev_bucketKeysQueries, numberOfBlocks, numberOfThreads);

	copyArrayToHost(bucketKeysData, dev_bucketKeysData, params.tables * params.N_data);
	copyArrayToHost(bucketKeysQueries, dev_bucketKeysQueries, params.tables * params.N_queries);

	/*printf("Bucket hashes data\n");
	for (int i = 0; i < params.tables * params.N_data; i++) {
		printf("[%d] = %d \n", i, bucketKeysData[i]);
	}*/

	printf("Finding bucket distribution \n");
	findBucketDistribution << <1, numberOfThreads >> > (params, dev_bucketKeysData, totalBucketCount, tableSize,dev_hashKeys); 
	waitForKernel(); 

	copyArrayToHost(hashKeys, dev_hashKeys, totalBucketCount);

	//int bucketSum = 0;
	//for (int i = 0; i < totalBucketCount; i++) {
	//	printf("[%d] = %d \n", i, hashKeys[i]);
	//	bucketSum += hashKeys[i];
	//}

	//printf("Buckets sum: %d\n", bucketSum);

	printf("Building bucket indexes \n");
	buildHashBucketIndexes << <1, params.tables >> > (params.N_data,tableSize, dev_hashKeys);
	waitForKernel();

	copyArrayToHost(hashKeys, dev_hashKeys, totalBucketCount); 
	
	/*for (int i = 0; i < totalBucketCount; i++) {
		printf("[%d] = %d \n", i , hashKeys[i]);
	}*/


	int* buckets = (int*)malloc(params.N_data * params.tables * sizeof(int));

	for (int i = 0; i < params.N_data * params.tables; i++) {
		buckets[i] = -1;
	}

	int* dev_buckets = mallocArray(buckets, params.N_data * params.tables, true);

	printf("Distribute points to buckets \n");
	runDistributePointsTobuckets(params, tableSize, dev_hashKeys, dev_bucketKeysData, dev_buckets);

	copyArrayToHost(buckets, dev_buckets, params.N_data * params.tables);

	printf("Bucket indexs: \n");

	/*for (int i = 0; i < params.N_data * params.tables; i++) {
		printf("[%d] = %d \n", i, buckets[i]);
	}*/


	int duplicateResultSize = numberOfBlocks * numberOfThreads * THREAD_QUEUE_SIZE;
	Point* resultsDuplicates = (Point*)malloc(duplicateResultSize * sizeof(Point));
	Point* dev_resultsDuplicates = mallocArray(resultsDuplicates, duplicateResultSize);

	printf("Running scan \n");
	scan << <numberOfBlocks, numberOfThreads>> > (params, dev_bucketKeysQueries, dev_hashKeys, dev_buckets, tableSize, dev_resultsDuplicates);
	waitForKernel(); 

	/*copyArrayToHost(resultsDuplicates, dev_resultsDuplicates, duplicateResultSize);

	for (int qi = 0; qi < 40 * 32; qi++) {
		printf("Query: %d\n", qi);
		for (int i = 0; i < THREAD_QUEUE_SIZE * WARPSIZE; i++) {
			printf("[%d] = (%d,%f)\n", i, resultsDuplicates[qi * THREAD_QUEUE_SIZE * WARPSIZE + i].ID, resultsDuplicates[qi * THREAD_QUEUE_SIZE * WARPSIZE + i].distance)
		}
	}*/


	Point* results = (Point*)malloc(params.N_queries * params.k * sizeof(Point));
	Point* dev_results = mallocArray(results, params.N_queries * params.k);

	int duplicateBlocks = ceil(params.N_queries / (float)numberOfThreads);
	printf("Removing duplicates from result\n");
	removeDuplicates << <duplicateBlocks, numberOfThreads >> > (params, dev_resultsDuplicates, dev_results);
	waitForKernel();

	copyArrayToHost(results, dev_results, params.N_queries * params.k);

	freeDeviceArray(dev_bucketKeysData);
	freeDeviceArray(dev_bucketKeysQueries);
	freeDeviceArray(dev_buckets);
	freeDeviceArray(dev_hashKeys);

	free(bucketKeysData);
	free(bucketKeysQueries);
	free(buckets);
	free(hashKeys);

	resetDevice(); 
	return results;
}