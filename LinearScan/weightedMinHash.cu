#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"point.h"
#include"simHash.cuh"
#include <iostream>
#include <random>
#include <cuda.h>
#include <bitset>
#include <math.h>
#include <time.h>
#include "constants.cuh"
#include "hammingDistanceScanner.cuh"
#include "launchHelper.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include "processingUtils.cuh"
#include "cudaHelpers.cuh"
#include "statistics.cuh"
#include "statisticsCpu.h"
#include <map>

#define DISTANCE_FUNCTION 2

__global__
void transformVectors(float* data, float* queries, int N_data, int N_queries, int dimensions, int* m_bounds) {
	transformData(data, queries, N_data, N_queries, dimensions, m_bounds);
}

__global__
void normalizeVectors(float* data, float* queries, int N_data, int N_queries, int dimensions) {
	transformToUnitVectors(queries, N_queries, dimensions);
	transformToUnitVectors(data, N_data, dimensions);
}

__global__
void preprocess(float* data, float* queries, int N_data, int N_queries, int dimensions,int* m_bounds, int* m_indexMapSize) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	
	// Find max
	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMax(&m_bounds[dim], ceil(data[i * dimensions + dim]));
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMax(&m_bounds[dim], ceil(queries[i * dimensions + dim]));
		}
	}

	__syncthreads();

	if (threadId == 0) {
		m_indexMapSize[0] = 0; 
		for (int i = 0; i < dimensions; i++) {
			m_indexMapSize[0] += m_bounds[i];
		}
	}

	__syncthreads();
}

__global__
void setupMapIndex(int* m_bounds, int* indexToComponentMap, int dimensions, int indexMapSize) {
	if (threadIdx.x == 0) { //TODO... We all know there is a smarter way to do this...
		int currentBound = 0;
		for (int i = 0; i < dimensions; i++) {
			int bound = currentBound + m_bounds[i];
			for (int j = currentBound; j < bound; j++) {
				if (j >= indexMapSize) {
					printf("j = %d and bounds[%d] = %d", j, i, m_bounds[i]);
				}
				else {
					indexToComponentMap[j] = i;
				}
			}

			m_bounds[i] = currentBound;
			currentBound = bound;
		}
	}
}

__inline__ __device__
float uniformRandom(curandState* state) {
	float val = curand_uniform(state); 
	return val; 
}

__inline__ __device__ 
bool isGreen(int* m_indexMap, int* m_bounds, float* data, float r, int i, int d) {
	int rIdx = r;
	int componentIdx = m_indexMap[rIdx];
	int m_bounds_val = m_bounds[componentIdx];
	float pointDI = data[i*d + componentIdx];
	return r <= m_bounds_val + pointDI; 

}


__global__
void sketchDataOneBit(float* data, int N_data, int dimensions, int sketchDim, int* m_indexMap, int* m_bounds, int M, int* seeds, bool* randomBitMap, unsigned char* sketchedData) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;

	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {

			for (int bitIndex = 0; bitIndex < 8; bitIndex++) {
				int seed = seeds[hashIdx * 8 + bitIndex];
				curandState s;
				curand_init(seed, 0, 10000, &s);
				bool red = true;
				int counter = 0;
				float r = 0;
				while (red) {
					float random = uniformRandom(&s);
					r = M * random;
					red = !isGreen(m_indexMap, m_bounds, data, r, i, dimensions);
					if (red) {
						char val = sketchedData[i * sketchDim + hashIdx];
						counter++;
					}
				}
				int bit = randomBitMap[counter];
				sketchedData[i * sketchDim + hashIdx] |= bit << bitIndex;
			}


		}
	}

	//if (threadId == 0) {
	//	for (int i = 0; i < sketchDim * N_data; i++) {
	//		for (int bitIndex = 7; bitIndex >= 0; bitIndex--)
	//			printf("%d", (sketchedData[i] >> bitIndex) & 1);
	//			//printf("%d \n", sketchedData[i]);
	//		printf("\n");
	//	}
	//}

}

__global__
void sketchData(float* data, int N_data, int dimensions, int sketchDim, int* m_indexMap, int* m_bounds, int M, int* seeds, unsigned char* sketchedData) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;

	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {
			int seed = seeds[hashIdx];
			sketchedData[i * sketchDim + hashIdx] = 0;
			curandState s;
			curand_init(seed, 0, 10000, &s);
			bool red = true;
			while (red) {
				float random = uniformRandom(&s);
				float r = M * random;
				red = !isGreen(m_indexMap, m_bounds, data, r, i, dimensions);
				if (red) {
					sketchedData[i * sketchDim + hashIdx]++;
				}
			}
		}
	}
}


__global__
void scan(float* originalData, float* originalQueries, int dimensions, unsigned char * data, unsigned char * queries, int sketchDim, int N_data, int N_query, int k, Point* result) {
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int queryIndex = warpId * dimensions;
	if (queryIndex < dimensions * N_query) {
		scanHammingDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, DISTANCE_FUNCTION,result);
	}
}

__global__ 
void bucketDistributionKernel(unsigned char* hashes, int hashesSize, int* res) {
	bucketDistribution(hashes, hashesSize, res);

}

bool* generateRandomVectors(int N, bool randomSeed = false) {

	// same seed 
	static bool* vectors = (bool*)malloc(N * sizeof(bool));
	std::default_random_engine generator;
	// different seeds
	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator

	std::uniform_int_distribution<int> distribution(0, 1); // Standard normal distribution.

	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		std::cout << vectors[i] << ",";
	}
	std::cout << std::endl; 
	return vectors;
}


Point* runWeightedMinHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int implementation) {
	int numberOfThreads = calculateThreadsLocal(N_query);
	int numberOfBlocks = calculateBlocksLocal(N_query);
	
	int dataSize = d * N_data;
	int querySize = d * N_query;
	int resultSize = k * N_query;
	int charSize = 255; 
	int m_indexMapSize = 0;

	//Setup data array.
	float* dev_data = mallocArray(data, dataSize, true);
	
	//Setup query array.
	float* dev_queries = mallocArray(queries, querySize, true);
	
	bool runOneBitMinHash = implementation != 4; 

	//Seeds
	int seedArrSize = runOneBitMinHash ? sketchedDim * 8: sketchedDim;
	int* seedArr = (int*)malloc(seedArrSize * sizeof(int));

	for (int i = 0; i < seedArrSize; i++) {
		seedArr[i] = i * 1234 + 92138;
	}

	int* dev_seedArr = mallocArray(seedArr, seedArrSize, true);
	bool* randomBitMap = generateRandomVectors(charSize); 
	bool* dev_randomBitMap = mallocArray(randomBitMap, charSize, true); 
	//Sketch arrays
	int sketchedDataSize = N_data * sketchedDim;
	unsigned char* sketchedData = (unsigned char*)malloc(sketchedDataSize * sizeof(unsigned char));
	unsigned char* dev_sketchedData = mallocArray(sketchedData, sketchedDataSize);


	int sketchedQueriesSize = N_query * sketchedDim;
	unsigned char* sketchedQueries = (unsigned char*)malloc(sketchedQueriesSize * sizeof(unsigned char));
	unsigned char* dev_sketchedQueries = mallocArray(sketchedQueries, sketchedQueriesSize);


	int* m_bounds = (int*)malloc(d * sizeof(int));
	int* dev_m_bounds = mallocArray(m_bounds, d);

	int* m_indexMapSizeArr = (int*)malloc(sizeof(int));
	int* dev_m_IndexMapSizeArr = mallocArray(m_indexMapSizeArr, 1);

	// Transform data
	clock_t before = clock();

	transformVectors << <1, numberOfThreads >> > (dev_data, dev_queries, N_data, N_query, d, dev_m_bounds);
	waitForKernel();

	normalizeVectors << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_queries, N_data, N_query, d);
	waitForKernel();

	preprocess << <1, numberOfThreads >> > (dev_data, dev_queries, N_data, N_query, d, dev_m_bounds, dev_m_IndexMapSizeArr);
	waitForKernel();
	clock_t time_lapsed = clock() - before;
	printf("Time to preprocess: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	copyArrayToHost(m_indexMapSizeArr, dev_m_IndexMapSizeArr, 1);
	m_indexMapSize = m_indexMapSizeArr[0];
	printf("Index map size: %d \n", m_indexMapSize);

	// Build maps
	int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
	int* dev_m_indexMap = mallocArray(m_IndexMap, m_indexMapSize);

	before = clock();
	setupMapIndex << <1, 1 >> > (dev_m_bounds, dev_m_indexMap, d, m_indexMapSize); //
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time to setup map: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	copyArrayToHost(m_IndexMap, dev_m_indexMap, m_indexMapSize);
	copyArrayToHost(m_bounds, dev_m_bounds, d);

	for (int i = 0; i < d; i++) {
		printf("%d ", m_bounds[i]);
	}
	printf("\n");
	for (int i = 0; i < m_indexMapSize; i++) {
		printf("%d ", m_IndexMap[i]);
	}
	printf("\n");

	printf("Starting sketch data \n");
	before = clock();

	if (runOneBitMinHash) {
		sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (dev_queries, N_query, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap,dev_sketchedQueries);
		waitForKernel();

		sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (dev_data, N_data, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap,dev_sketchedData);
		waitForKernel();
	}
	else {
		sketchData << <numberOfBlocks, numberOfThreads >> > (dev_queries, N_query, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_sketchedQueries);
		waitForKernel();

		sketchData << <numberOfBlocks, numberOfThreads >> > (dev_data, N_data, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_sketchedData);
		waitForKernel();
	}
	
	time_lapsed = clock() - before;
	printf("Time to hash on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	
	printf("Done sketching \nStarting scan \n");
	int bucket_results_size = 255; 
	int* bucket_results = (int*)malloc(bucket_results_size * sizeof(int));
	int* bucket_results_dev = mallocArray(bucket_results, bucket_results_size);
	bucketDistributionKernel << <1, numberOfThreads >> > (dev_sketchedData, sketchedDataSize, bucket_results_dev);
	waitForKernel(); 


	copyArrayToHost(sketchedData, dev_sketchedData, sketchedDataSize);
	copyArrayToHost(sketchedQueries, dev_sketchedQueries, sketchedQueriesSize);

	copyArrayToHost(bucket_results, bucket_results_dev, bucket_results_size); 
	for (int i = 0; i < bucket_results_size; i++) {
		if (bucket_results[i] != 0) {
			printf("[%d] = %d \n", i, bucket_results[i]);
		}
	}

	//std::map<std::string, int> m = bucketDistributionFullKey(sketchedData, sketchedDataSize, sketchedDim); 

	//for (std::map<std::string,int>::iterator it = m.begin(); it != m.end(); ++it) {
	//	std::cout << it->first << " " << it->second << std::endl; 
	//}

	// Do linear scan
	Point* results = (Point*)malloc(resultSize * sizeof(Point));
	Point* dev_results = mallocArray(results, resultSize);

	before = clock();
	scan << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_queries, d, dev_sketchedData, dev_sketchedQueries, sketchedDim, N_data, N_query, k, dev_results);
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time for scanning: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	copyArrayToHost(results, dev_results, resultSize);

	printf("Done with scan \n");
	//Close
	freeDeviceArray(dev_data);
	freeDeviceArray(dev_queries);
	freeDeviceArray(dev_sketchedData);
	freeDeviceArray(dev_sketchedQueries);
	freeDeviceArray(dev_m_bounds);
	freeDeviceArray(dev_results);
	free(sketchedData);
	free(sketchedQueries);
	free(m_bounds);

	resetDevice();

	return results;
}