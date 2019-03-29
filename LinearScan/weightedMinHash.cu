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
#include "sketchedDistanceScanners.cuh"
#include "launchHelper.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include "processingUtils.cuh"
#include "cudaHelpers.cuh"
#include "statistics.cuh"
#include "statisticsCpu.h"
#include <map>
#include "launchDTO.h"

#define DISTANCE_FUNCTION 2

clock_t before;
clock_t time_lapsed;

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
void preprocess(float* data, float* queries, int N_data, int N_queries, int dimensions, int* m_bounds, int* m_indexMapSize) {
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


template<class T> __global__
void scan(float* originalData, float* originalQueries, int dimensions, T * data, T * queries, int sketchDim, int N_data, int N_query, int k, Point* result) {
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int queryIndex = warpId * dimensions;
	if (queryIndex < dimensions * N_query) {
		scanJaccardDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, DISTANCE_FUNCTION, result);
	}
}

__global__
void bucketDistributionKernel(unsigned char* hashes, int hashesSize, int* res) {
	bucketDistribution(hashes, hashesSize, res);
}

template<class T>
void minHashPreprocessing(LaunchDTO<T> launchDTO, int* dev_m_bounds, int* dev_m_IndexMapSizeArr, int numberOfBlocks, int numberOfThreads) {

	// Transform data
	before = clock();

	transformVectors << <1, numberOfThreads >> > (launchDTO.data, launchDTO.queries, launchDTO.N_data, launchDTO.N_queries, launchDTO.dimensions, dev_m_bounds);
	waitForKernel();

	normalizeVectors << <numberOfBlocks, numberOfThreads >> > (launchDTO.data, launchDTO.queries, launchDTO.N_data, launchDTO.N_queries, launchDTO.dimensions);
	waitForKernel();

	preprocess << <1, numberOfThreads >> > (launchDTO.data, launchDTO.queries, launchDTO.N_data, launchDTO.N_queries, launchDTO.dimensions, dev_m_bounds, dev_m_IndexMapSizeArr);
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time to preprocess: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
}

template<class T>
void runBucketStatistics(LaunchDTO<T> launchDTO, int numberOfThreads) {
	int bucket_results_size = 255;
	int* bucket_results = (int*)malloc(bucket_results_size * sizeof(int));
	int* bucket_results_dev = mallocArray(bucket_results, bucket_results_size);
	bucketDistributionKernel << <1, numberOfThreads >> > (launchDTO.sketchedData, launchDTO.sketchedDataSize, bucket_results_dev);
	waitForKernel();

	copyArrayToHost(bucket_results, bucket_results_dev, bucket_results_size);
	for (int i = 0; i < bucket_results_size; i++) {
		if (bucket_results[i] != 0) {
			printf("[%d] = %d \n", i, bucket_results[i]);
		}
	}
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

	bool* dev_vectors = mallocArray(vectors, N, true);
	return dev_vectors;
}

int* createSeedArr(int seedArrSize) {
	int* seedArr = (int*)malloc(seedArrSize * sizeof(int));

	for (int i = 0; i < seedArrSize; i++) {
		seedArr[i] = i * 1234 + 92138;
	}

	int* dev_seedArr = mallocArray(seedArr, seedArrSize, true);

	return dev_seedArr;
}

int getIndexMapSize(int* m_indexMapSizeArr, int* dev_m_IndexMapSizeArr) {
	copyArrayToHost(m_indexMapSizeArr, dev_m_IndexMapSizeArr, 1);
	int m_indexMapSize = m_indexMapSizeArr[0];
	printf("Index map size: %d \n", m_indexMapSize);
	return m_indexMapSize;
}

int* createIndexMap(int indexMapSize) {

	int* m_IndexMap = (int*)malloc(indexMapSize * sizeof(int));
	int* dev_m_indexMap = mallocArray(m_IndexMap, indexMapSize);
	return dev_m_indexMap;
}


void printIndexMaps(int* dev_m_indexMap, int* dev_m_bounds, int d, int m_indexMapSize) {

	int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
	int* m_bounds = (int*)malloc(d * sizeof(int));
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

	free(m_IndexMap);
	free(m_bounds);
}

void setupMaps(int* dev_m_indexMap, int* dev_m_bounds, int d, int m_indexMapSize, bool print = false) {

	before = clock();
	setupMapIndex << <1, 1 >> > (dev_m_bounds, dev_m_indexMap, d, m_indexMapSize); //
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time to setup map: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	if (print)printIndexMaps(dev_m_indexMap, dev_m_bounds, d, m_indexMapSize);
}

template <class T>
Point* runScan(LaunchDTO<T> launchDTO, int numberOfBlocks, int numberOfThreads) {

	Point* results = (Point*)malloc(launchDTO.resultSize * sizeof(Point));

	before = clock();
	scan << <numberOfBlocks, numberOfThreads >> > (launchDTO.data, launchDTO.queries, launchDTO.dimensions, launchDTO.sketchedData, launchDTO.sketchedQueries, launchDTO.sketchDim, launchDTO.N_data, launchDTO.N_queries, launchDTO.k, launchDTO.results);
	waitForKernel();
	time_lapsed = clock() - before;
	printf("Time for scanning: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	copyArrayToHost(results, launchDTO.results, launchDTO.resultSize);

	printf("Done with scan \n");

	return results; 
}

template <class T>
void cleanup(LaunchDTO<T> launchDTO, int* dev_m_bounds, int* m_bounds, bool* dev_randomBitMap) {
	freeDeviceArray(launchDTO.data);
	freeDeviceArray(launchDTO.queries);
	freeDeviceArray(launchDTO.sketchedData);
	freeDeviceArray(launchDTO.sketchedQueries);
	freeDeviceArray(dev_m_bounds);
	freeDeviceArray(launchDTO.results);
	freeDeviceArray(dev_randomBitMap);
	free(m_bounds);
}

template <class T>
LaunchDTO<T> setupLaunchDTO(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {
	LaunchDTO<T> launchDTO;

	launchDTO.k = k;
	launchDTO.dimensions = d;
	launchDTO.sketchDim = sketchedDim;
	launchDTO.N_data = N_data;
	launchDTO.N_queries = N_query;
	launchDTO.dataSize = N_data * d;
	launchDTO.querySize = N_query * d;
	launchDTO.resultSize = N_query * k;
	launchDTO.sketchedDataSize = N_data * sketchedDim;
	launchDTO.sketchedQueriesSize = N_query * sketchedDim;
	Point* results = (Point*)malloc(launchDTO.resultSize * sizeof(Point));
	unsigned char* sketchedData;
	unsigned char* sketchedQueries;

	launchDTO.data = mallocArray(data, launchDTO.dataSize, true);
	launchDTO.queries = mallocArray(queries, launchDTO.querySize, true);
	launchDTO.results = mallocArray(results, launchDTO.resultSize);
	launchDTO.sketchedData = mallocArray(sketchedData, launchDTO.sketchedDataSize);
	launchDTO.sketchedQueries = mallocArray(sketchedQueries, launchDTO.sketchedQueriesSize);
	free(results);
	return launchDTO;
}

Point* runWeightedMinHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int implementation) {
	int numberOfThreads = calculateThreadsLocal(N_query);
	int numberOfBlocks = calculateBlocksLocal(N_query);
	clock_t before = clock();
	clock_t time_lapsed = clock();
	int charSize = 255;

	LaunchDTO<unsigned char> launchDTO = setupLaunchDTO<unsigned char>(k, d, sketchedDim, N_query, N_data, data, queries);
	printf("Done setting up DTO \n");

	//Setup query array.

	bool runOneBitMinHash = implementation != 4;

	int* dev_seedArr = createSeedArr(runOneBitMinHash ? sketchedDim * 8 : sketchedDim);

	bool* dev_randomBitMap = generateRandomVectors(charSize);

	int* m_bounds = (int*)malloc(d * sizeof(int));
	int* dev_m_bounds = mallocArray(m_bounds, d);

	int* m_indexMapSizeArr = (int*)malloc(sizeof(int));
	int* dev_m_IndexMapSizeArr = mallocArray(m_indexMapSizeArr, 1);

	//Transform, Normalize, Maps
	minHashPreprocessing(launchDTO, dev_m_bounds, dev_m_IndexMapSizeArr, numberOfBlocks, numberOfThreads);

	int m_indexMapSize = getIndexMapSize(m_indexMapSizeArr, dev_m_IndexMapSizeArr);
	printf("Index map size: %d \n", m_indexMapSize);

	// Build maps
	int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
	int* dev_m_indexMap = createIndexMap(m_indexMapSize); //mallocArray(m_IndexMap, m_indexMapSize);

	//Finalize maps
	setupMaps(dev_m_indexMap, dev_m_bounds, d, m_indexMapSize, true); 


	printf("Starting sketch data \n");
	before = clock();

	if (runOneBitMinHash) {
		sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions, launchDTO.sketchDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap, launchDTO.sketchedQueries);
		waitForKernel();

		sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (launchDTO.data, launchDTO.N_data, launchDTO.dimensions, launchDTO.sketchDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap, launchDTO.sketchedData);
		waitForKernel();
	}
	else {
		sketchData << <numberOfBlocks, numberOfThreads >> > (launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions, launchDTO.sketchDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, launchDTO.sketchedQueries);
		waitForKernel();

		sketchData << <numberOfBlocks, numberOfThreads >> > (launchDTO.data, launchDTO.N_data, launchDTO.dimensions, launchDTO.sketchDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, launchDTO.sketchedData);
		waitForKernel();
	}

	time_lapsed = clock() - before;
	printf("Time to hash on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	
	runBucketStatistics(launchDTO, numberOfThreads); 
	
	printf("Done sketching \nStarting scan \n");
	Point* results = runScan(launchDTO, numberOfBlocks, numberOfThreads);

	//Close
	cleanup(launchDTO, dev_m_bounds, m_bounds, dev_randomBitMap);

	resetDevice();

	return results;
}