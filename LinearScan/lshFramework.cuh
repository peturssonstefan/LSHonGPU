#pragma once

#include "point.h"
#include "launchDTO.h"
#include "simHash.cuh"
#include "randomVectorGenerator.h" 
#include "device_launch_parameters.h"
#include "cudaHelpers.cuh"
#include <bitset>
#include <math.h>
#include <cuda.h>
#include "weightedMinHash.cuh"
#include "simHashJL.cuh"
#include "crossPolyTope.cuh"

template<class T>
void runSketchJL(LaunchDTO<T> params) {
	simHashJl::runGenericJL(params); 
}


template<class T, class K>
void runSketchSimHash(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int numberOfBlocks, int numberOfThreads, bool isHashKeys) {
	float randomVectorsSize = isHashKeys ? lshParams.tables * params.dimensions * lshParams.bucketKeyBits : params.sketchDim * params.dimensions * SKETCH_COMP_SIZE;
	
	float* randomVectors = (float*)malloc(randomVectorsSize * sizeof(float));
	generateRandomVectors(randomVectors,randomVectorsSize);
	float* dev_randomVectors = mallocArray(randomVectors, randomVectorsSize, true);

	if (isHashKeys) {
		simHash::sketchDataGeneric << <numberOfBlocks, numberOfThreads >> > (params.data, dev_randomVectors, params.N_data, params.dimensions, lshParams.tables, lshParams.bucketKeyBits, lshParams.dataKeys);
		waitForKernel();
		simHash::sketchDataGeneric << <numberOfBlocks, numberOfThreads >> > (params.queries, dev_randomVectors, params.N_queries, params.dimensions, lshParams.tables, lshParams.bucketKeyBits, lshParams.queryKeys);
		waitForKernel();
	}
	else {
		simHash::sketchDataGeneric << <numberOfBlocks, numberOfThreads >> > (params.data, dev_randomVectors, params.N_data, params.dimensions, params.sketchDim, SKETCH_COMP_SIZE, params.sketchedData);
		waitForKernel();
		simHash::sketchDataGeneric << <numberOfBlocks, numberOfThreads >> > (params.queries, dev_randomVectors, params.N_queries, params.dimensions, params.sketchDim, SKETCH_COMP_SIZE, params.sketchedQueries);
		waitForKernel();
	}
	
	freeDeviceArray(dev_randomVectors);
	free(randomVectors);
}

template<class T, class K>
void runSketchMinHash(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, Result& res, int numberOfBlocks, int numberOfThreads, bool isHashKeys) {

	if (isHashKeys) {
		bool oneBitMinHash = lshParams.keyImplementation != 4; 
		weightedMinHash::weightedMinHashGeneric(params, res, lshParams.dataKeys, lshParams.queryKeys, lshParams.tables, lshParams.bucketKeyBits, oneBitMinHash);
	}
	else {
		bool oneBitMinHash = params.implementation != 4;
		weightedMinHash::weightedMinHashGeneric(params, res, params.sketchedData, params.sketchedQueries, params.sketchDim, SKETCH_COMP_SIZE, oneBitMinHash);
	}
}


template <class T, class K>
void generateHashes(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, Result& res, int numberOfBlocks, int numberOfThreads, bool isHashKeys) {
	int implementation = isHashKeys ? lshParams.keyImplementation : params.implementation;
	switch (implementation) {
	
	case 2: break;  //Run without sketching; 
	case 3:
		printf("Using Simhash to %s \n", isHashKeys ? "generate keys" : "to sketch");
		runSketchSimHash(params, lshParams, numberOfBlocks, numberOfThreads, isHashKeys);
		break; 
	case 4: 
		printf("Using Weighted Minhash to %s \n", isHashKeys ? "generate keys" : "to sketch");
		runSketchMinHash(params, lshParams, res, numberOfBlocks, numberOfThreads, isHashKeys); 
		break; 
	case 5:
		printf("Using 1 Bit Weighted Minhash to %s \n", isHashKeys ? "generate keys" : "to sketch");
		runSketchMinHash(params, lshParams, res, numberOfBlocks, numberOfThreads, isHashKeys);
		break; 
	case 6: 
		printf("Using Johnson Lindenstrauss to sketch \n");
		if (isHashKeys) {
			printf("JL is not suited as LSH hashkey \n");
			resetDevice(); 
			exit(-1);
		}
		runSketchJL(params);
		break;
	case 7: 
		if (isHashKeys) {
			printf("Using CrossPoly as hash key \n"); 
			crosspoly::crossPolyTopeHashing(params, lshParams, res, numberOfBlocks, numberOfThreads);
			break;
		}
		else {
			printf("Cross Poly is not suited as sketching algorithm \n");
			resetDevice();
			exit(-1); 
		}
	}
}

template<class T, class K> __global__ 
void findBucketDistribution(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int bucketCount, int* hashKeys) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	
	for (int i = threadId; i < params.N_data; i += threadCount) {
		for (int tableIdx = 0; tableIdx < lshParams.tables; tableIdx++) {
			int hash = lshParams.dataKeys[i*lshParams.tables + tableIdx];
			atomicAdd(&hashKeys[tableIdx * lshParams.tableSize + hash], 1);
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

template<class T, class K> __global__
void distributePointsToBuckets(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int* hashKeys, int* bucketCounters, int* buckets) {
	int threadId = (blockIdx.x * gridDim.x) + threadIdx.x;
	int warpId = threadId / WARPSIZE;
	int lane = threadId % WARPSIZE;
	for (int i = lane; i < params.N_data; i+=WARPSIZE) {
		int hash = lshParams.dataKeys[i * lshParams.tables + warpId];
		int bucketIdx = atomicAdd(&bucketCounters[warpId * lshParams.tableSize + hash], 1);
		int bucketStart = hashKeys[warpId * lshParams.tableSize + hash];

		buckets[bucketStart + bucketIdx] = i;
	}
}

template <class T, class K> __global__ 
void scan(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int* dev_hashKeys, int* dev_buckets, Point* result) {

	Point threadQueue[THREAD_QUEUE_SIZE];

	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int warpId = threadId / WARPSIZE;
	float maxKDistance = (float)INT_MAX;
	int lane = threadId % WARPSIZE;
	int warpQueueSize = params.k / WARPSIZE;
	int localMaxKDistanceIdx = THREAD_QUEUE_SIZE - warpQueueSize;
	int candidateSetSize = THREAD_QUEUE_SIZE - warpQueueSize;
	int queryHashIdx = warpId * lshParams.tables;
	int querySketchedIdx = warpId * params.sketchDim;
	int queryIdx = warpId * params.dimensions; 
	int resultIdx = warpId * THREAD_QUEUE_SIZE * WARPSIZE;
	float magnitudeQuery = 0; 
	Point swapPoint;
	Parameters sortParameters;
	sortParameters.lane = lane; 

	bool sketchTypeOneBit = params.implementation != 4;
	int similarityDivisor = sketchTypeOneBit ? params.sketchDim * SKETCH_COMP_SIZE : params.sketchDim;

	int queuePosition = 0;
	
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		threadQueue[i] = createPoint(-1, maxKDistance);
	}

	for (int dim = 0; dim < params.dimensions; dim++) {
		magnitudeQuery += params.queries[queryIdx + dim] * params.queries[queryIdx + dim];
	}

	magnitudeQuery = sqrt(magnitudeQuery); 

	for (int tableIdx = 0; tableIdx < lshParams.tables; tableIdx++) {
		
		// Find bucket place
		unsigned int queryHashKey = lshParams.queryKeys[queryHashIdx + tableIdx];
		//if(lane == 0) printf("Query: %d, QueryHash: %d \n", warpId, queryHashKey); 
		int bucketStart = dev_hashKeys[lshParams.tableSize * tableIdx + queryHashKey];
		int bucketEnd = queryHashKey >= lshParams.tableSize - 1 ? tableIdx * params.N_data + params.N_data : dev_hashKeys[tableIdx * lshParams.tableSize + queryHashKey + 1];
		//if (lane == 0) printf("Query: %d, Bucket start: %d \n", warpId, bucketStart);
		//scan bucket
		for (int bucketIdx = bucketStart + lane; bucketIdx < bucketEnd; bucketIdx += WARPSIZE) {
			int dataIdx = dev_buckets[bucketIdx];

			float distance = lshParams.runWithSketchedData ?
				  runSketchedDistanceFunction(params.implementation, &params.sketchedData[dataIdx * params.sketchDim], &params.sketchedQueries[querySketchedIdx], params.sketchDim, similarityDivisor)
				: runDistanceFunction(params.distanceFunc, &params.data[dataIdx * params.dimensions], &params.queries[queryIdx], params.dimensions, magnitudeQuery);
			
			Point point = createPoint(dataIdx, distance);

			if (WITH_TQ_OR_BUFFER) {
				//run TQ
				for (int j = candidateSetSize - 1; j >= 0; j--) { // simple sorting.
					if (point.distance < threadQueue[j].distance) {
						swapPoint = threadQueue[j];
						threadQueue[j] = point;
						point = swapPoint;
					}
				}

				//Verify that head of thread queue is not smaller than biggest k distance.
				if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
					startSort(threadQueue, swapPoint, sortParameters);
					maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
				}
			}
			else {
				//run buffer
				if (point.distance < maxKDistance || same(point, maxKDistance)) {
					threadQueue[queuePosition++] = point;
				}

				if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
					startSort(threadQueue, swapPoint, sortParameters);
					maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
					//printQueue(threadQueue);
					queuePosition = 0;
				}
			}

		}

		 //Might need at sort here
		//if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance)) {
		//	startSort(threadQueue, swapPoint, sortParameters);
		//	maxKDistance = broadCastMaxK(threadQueue[localMaxKDistanceIdx].distance);
		//}
	}


	if (lshParams.runWithSketchedData) {
		candidateSetScan(params.data, &params.queries[queryIdx], params.dimensions, threadQueue, params.k, params.distanceFunc);
	}
	
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

template<class T, class K>
void runDistributePointsTobuckets(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int* dev_hashKeys, int* dev_buckets) {
	int* bucketsCounters = (int*)malloc(lshParams.tableSize * lshParams.tables * sizeof(int));

	for (int i = 0; i < lshParams.tableSize * lshParams.tables; i++) {
		bucketsCounters[i] = 0;
	}
	int* dev_bucketCounters = mallocArray(bucketsCounters, lshParams.tableSize * lshParams.tables, true);

	distributePointsToBuckets << <1, lshParams.tables * WARPSIZE >> > (params, lshParams, dev_hashKeys, dev_bucketCounters, dev_buckets);
	waitForKernel();

	freeDeviceArray(dev_bucketCounters);
	free(bucketsCounters);
}

template <class T, class K>
void cleanup(LaunchDTO<T> params, LshLaunchDTO<K> lshParams) {
	freeDeviceArray(params.data);
	freeDeviceArray(params.queries);
	freeDeviceArray(params.sketchedData);
	freeDeviceArray(params.sketchedQueries);
	freeDeviceArray(params.results);
	freeDeviceArray(lshParams.dataKeys);
	freeDeviceArray(lshParams.queryKeys);
}

template<class T, class K>
Result runLsh(LaunchDTO<T> params, LshLaunchDTO<K> lshParams) {
	Result res;
	res.setupResult(params.N_queries, params.k); 
	int totalBucketCount = lshParams.tableSize * lshParams.tables;
	int* hashKeys = (int*)malloc(totalBucketCount * sizeof(int));
	for (int i = 0; i < totalBucketCount; i++) {
		hashKeys[i] = 0; 
	}

	printf("tableSize: %d \n", lshParams.tableSize);

	int* dev_hashKeys = mallocArray(hashKeys, totalBucketCount, true); 
	int numberOfThreads = calculateThreadsLocal(params.N_queries);
	int numberOfBlocks = calculateBlocksLocal(params.N_queries);

	printf("Building key hashes \n");
	time_t before = clock(); 
	generateHashes(params, lshParams, res, numberOfBlocks, numberOfThreads, true);

	K* bucketKeysData = (K*)malloc(lshParams.tables * params.N_data * sizeof(K)); 
	K* bucketKeysQueries = (K*)malloc(lshParams.tables * params.N_queries * sizeof(K));
	copyArrayToHost(bucketKeysData, lshParams.dataKeys, lshParams.tables * params.N_data);
	copyArrayToHost(bucketKeysQueries, lshParams.queryKeys, lshParams.tables * params.N_queries);

	/*printf("Bucket hashes data\n");
	for (int i = 0; i < lshParams.tables * params.N_data; i++) {
		printf("[%u] = %d \n", i, bucketKeysData[i]);
	}*/



	printf("Finding bucket distribution \n");
	findBucketDistribution << <1, numberOfThreads >> > (params, lshParams, totalBucketCount, dev_hashKeys); 
	waitForKernel(); 

	copyArrayToHost(hashKeys, dev_hashKeys, totalBucketCount);

	/*int bucketSum = 0;
	for (int i = 0; i < totalBucketCount; i++) {
		printf("[%d] = %d \n", i, hashKeys[i]);
		bucketSum += hashKeys[i];
	}

	printf("Buckets sum: %d\n", bucketSum);*/


	printf("Building bucket indexes \n");
	buildHashBucketIndexes << <1, lshParams.tables >> > (params.N_data, lshParams.tableSize, dev_hashKeys);
	waitForKernel();

	copyArrayToHost(hashKeys, dev_hashKeys, totalBucketCount); 
	
	//for (int i = 0; i < totalBucketCount; i++) {
	//	printf("[%d] = %d \n", i , hashKeys[i]);
	//}


	int* buckets = (int*)malloc(params.N_data * lshParams.tables * sizeof(int));

	for (int i = 0; i < params.N_data * lshParams.tables; i++) {
		buckets[i] = -1;
	}

	int* dev_buckets = mallocArray(buckets, params.N_data * lshParams.tables, true);

	printf("Distribute points to buckets \n");
	runDistributePointsTobuckets(params, lshParams, dev_hashKeys, dev_buckets);

	copyArrayToHost(buckets, dev_buckets, params.N_data * lshParams.tables);

	//printf("Bucket indexs: \n");

	//for (int i = 0; i < params.N_data * lshParams.tables; i++) {
	//	printf("[%d] = %d \n", i, buckets[i]);
	//}


	//sketch data
	generateHashes(params, lshParams, res, numberOfBlocks, numberOfThreads, false);

	/*printf("Malloc \n");
	T* dataSketch = (T*)malloc(params.sketchedDataSize * sizeof(T));
	printf("Copy \n");
	copyArrayToHost(dataSketch, params.sketchedData, params.sketchedDataSize);

	printf("Printing \n");
	for (int i = 0; i < params.N_data; i++) {
		printf("DataIdx: %d \n", i); 
		for(int j = 0; j < params.sketchDim; j++)
			printf("%d ", dataSketch[i * params.sketchDim + j]);

		printf("\n"); 
	}*/
	time_t time_lapsed = clock() - before; 
	res.constructionTime = res.calcTime(time_lapsed) - res.preprocessTime; //Some hash functions include preprocessing of the data. 
	printf("Time to preprocess: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	int duplicateResultSize = numberOfBlocks * numberOfThreads * THREAD_QUEUE_SIZE;
	Point* resultsDuplicates = (Point*)malloc(duplicateResultSize * sizeof(Point));
	Point* dev_resultsDuplicates = mallocArray(resultsDuplicates, duplicateResultSize);
	
	printf("Running scan \n");
	before = clock(); 
	size_t free_byte;
	size_t total_byte;
	cudaMemGetInfo(&free_byte,&total_byte);
	double free_byte_double = (double)free_byte; 
	double total_byte_double = (double)total_byte; 
	double used_bytes = total_byte_double - free_byte_double; 
	printf("Free: %f, Total: %f, used %f \n", (free_byte_double/1048576), (total_byte_double / 1048576), (used_bytes/1048576));	
	scan << <numberOfBlocks * 2, 512>> > (params, lshParams, dev_hashKeys, dev_buckets, dev_resultsDuplicates);
	waitForKernel(); 
	time_lapsed = clock() - before;
	res.scanTime = res.calcTime(time_lapsed); 
	printf("Time to calculate distance on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	/*copyArrayToHost(resultsDuplicates, dev_resultsDuplicates, duplicateResultSize);

	for (int qi = 0; qi < 40 * 32; qi++) {
		printf("Query: %d\n", qi);
		for (int i = 0; i < THREAD_QUEUE_SIZE * WARPSIZE; i++) {
			printf("[%d] = (%d,%f)\n", i, resultsDuplicates[qi * THREAD_QUEUE_SIZE * WARPSIZE + i].ID, resultsDuplicates[qi * THREAD_QUEUE_SIZE * WARPSIZE + i].distance)
		}
	}*/


	Point* results = (Point*)malloc(params.N_queries * params.k * sizeof(Point));
	Point* dev_results = mallocArray(results, params.N_queries * params.k);
	before = clock(); 
	int duplicateBlocks = ceil(params.N_queries / (float)numberOfThreads);
	printf("Removing duplicates from result\n");
	removeDuplicates << <duplicateBlocks, numberOfThreads >> > (params, dev_resultsDuplicates, dev_results);
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time to remove duplicates: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	copyArrayToHost(results, dev_results, params.N_queries * params.k);
	res.copyResultPoints(results, params.N_queries, params.k); 
	cleanup(params, lshParams); 
	freeDeviceArray(dev_buckets);
	freeDeviceArray(dev_hashKeys);

	free(buckets);
	free(hashKeys);

	resetDevice(); 
	return res;
}
