#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
#include "launchHelper.cuh"

namespace weightedMinHash {



#define DISTANCE_FUNCTION 2

	template <class T> __global__
		void transformVectors(LaunchDTO<T> launchDTO, int* m_bounds) {
		transformData(launchDTO.data, launchDTO.queries, launchDTO.N_data, launchDTO.N_queries, launchDTO.dimensions, m_bounds);
	}

	template <class T> __global__
		void normalizeVectors(LaunchDTO<T> launchDTO) {
		transformToUnitVectors(launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions);
		transformToUnitVectors(launchDTO.data, launchDTO.N_data, launchDTO.dimensions);
	}

	template <class T>__global__
		void preprocess(LaunchDTO<T> launchDTO, int* m_bounds, int* m_indexMapSize) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		int totalThreads = blockDim.x * gridDim.x;

		// Find max
		for (int i = threadId; i < launchDTO.N_data; i += totalThreads) {
			for (int dim = 0; dim < launchDTO.dimensions; dim++) {
				atomicMax(&m_bounds[dim], ceil(launchDTO.data[i * launchDTO.dimensions + dim]));
			}
		}

		for (int i = threadId; i < launchDTO.N_queries; i += totalThreads) {
			for (int dim = 0; dim < launchDTO.dimensions; dim++) {
				atomicMax(&m_bounds[dim], ceil(launchDTO.queries[i * launchDTO.dimensions + dim]));
			}
		}

		__syncthreads();

		if (threadId == 0) {
			m_indexMapSize[0] = 0;
			for (int i = 0; i < launchDTO.dimensions; i++) {
				m_indexMapSize[0] += m_bounds[i];
			}
		}

		__syncthreads();
	}

	__global__
		void setupMapIndex(int* m_bounds, int* indexToComponentMap, int dimensions, int indexMapSize) {
		if (threadIdx.x == 0) {
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


	template<class T, class K>__global__
		void sketchDataOneBit(LaunchDTO<T> launchDTO, float* data, int N_data, int sketchedDim, int hashBits, int* m_indexMap, int* m_bounds, int M, int* seeds, bool* randomBitMap, K* sketchedData) {
		int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
		int totalThreads = blockDim.x * gridDim.x;

		for (int i = threadId; i < N_data; i += totalThreads) {
			for (int hashIdx = 0; hashIdx < sketchedDim; hashIdx++) {
				unsigned int sketch = 0;
				for (int bitIndex = 0; bitIndex < hashBits; bitIndex++) {
					int seed = seeds[hashIdx * hashBits + bitIndex];
					curandState s;
					curand_init(seed, 0, 10000, &s);
					bool red = true;
					int counter = 0;
					float r = 0;
					while (red) {
						float random = uniformRandom(&s);
						r = M * random;
						red = !isGreen(m_indexMap, m_bounds, data, r, i, launchDTO.dimensions);
						if (red) {
							counter++;
						}
					}
					unsigned int bit = randomBitMap[counter];
					sketch |= bit << bitIndex;;
				}

				sketchedData[i * sketchedDim + hashIdx] = sketch;
			}
		}

	}



	template<class T, class K>__global__
		void sketchDataOriginal(LaunchDTO<T> launchDTO, float* data, int N_data, int sketchedDim, int* m_indexMap, int* m_bounds, int M, int* seeds, K* sketchedData) {
		int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
		int totalThreads = blockDim.x * gridDim.x;

		for (int i = threadId; i < N_data; i += totalThreads) {
			for (int hashIdx = 0; hashIdx < sketchedDim; hashIdx++) {
				int seed = seeds[hashIdx];
				sketchedData[i * sketchedDim + hashIdx] = 0;
				curandState s;
				curand_init(seed, 0, 10000, &s);
				bool red = true;
				while (red) {
					float random = uniformRandom(&s);
					float r = M * random;
					red = !isGreen(m_indexMap, m_bounds, data, r, i, launchDTO.dimensions);
					if (red) {
						sketchedData[i * sketchedDim + hashIdx]++;
					}
				}
			}
		}
	}

	template<class T, class K>
	void sketchData(LaunchDTO<T> launchDTO, int sketchedDim, K* sketchedData, K* sketchedQueries, int* dev_m_indexMap, int* dev_m_bounds, int m_indexMapSize, int* dev_seedArr, bool oneBit, int hashBits, bool* dev_randomBitMap, int numberOfBlocks, int numberOfThreads) {
		if (oneBit) {
			sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.queries, launchDTO.N_queries, sketchedDim, hashBits, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap, sketchedQueries);
			waitForKernel();

			sketchDataOneBit << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.data, launchDTO.N_data, sketchedDim, hashBits, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_randomBitMap, sketchedData);
			waitForKernel();
		}
		else {
			sketchDataOriginal << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.queries, launchDTO.N_queries, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, sketchedQueries);
			waitForKernel();

			sketchDataOriginal << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.data, launchDTO.N_data, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, sketchedData);
			waitForKernel();
		}
	}

	template<class T> __global__
		void scan(float* originalData, float* originalQueries, int dimensions, T * data, T * queries, int sketchDim, int N_data, int N_query, int k, int implementation, Point* result) {

		int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
		int queryIndex = warpId * dimensions;
		if (queryIndex < dimensions * N_query) {
			scanJaccardDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, DISTANCE_FUNCTION, implementation, result);
		}
	}

	template<class T> __global__
		void bucketDistributionKernel(T* hashes, int hashesSize, int* res) {
		bucketDistribution(hashes, hashesSize, res);
	}

	template<class T>
	void minHashPreprocessing(LaunchDTO<T> launchDTO, int* dev_m_bounds, int* dev_m_IndexMapSizeArr, int numberOfBlocks, int numberOfThreads) {
		clock_t before;
		clock_t time_lapsed;

		// Transform data
		before = clock();

		transformVectors << <1, numberOfThreads >> > (launchDTO, dev_m_bounds);
		waitForKernel();

		normalizeVectors << <numberOfBlocks, numberOfThreads >> > (launchDTO);
		waitForKernel();

		preprocess << <1, numberOfThreads >> > (launchDTO, dev_m_bounds, dev_m_IndexMapSizeArr);
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

	bool* generateRandomBoolVectors(int N, bool randomSeed = false) {

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
		clock_t before;
		clock_t time_lapsed;

		before = clock();
		setupMapIndex << <1, 1 >> > (dev_m_bounds, dev_m_indexMap, d, m_indexMapSize);
		waitForKernel();
		time_lapsed = clock() - before;
		printf("Time to setup map: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
		if (print)printIndexMaps(dev_m_indexMap, dev_m_bounds, d, m_indexMapSize);
	}

	template <class T>
	Point* runScan(LaunchDTO<T> launchDTO, int numberOfBlocks, int numberOfThreads) {
		clock_t before;
		clock_t time_lapsed;

		Point* results = (Point*)malloc(launchDTO.resultSize * sizeof(Point));

		before = clock();
		scan << <numberOfBlocks, numberOfThreads >> > (launchDTO.data, launchDTO.queries, launchDTO.dimensions, launchDTO.sketchedData, launchDTO.sketchedQueries, launchDTO.sketchDim, launchDTO.N_data, launchDTO.N_queries, launchDTO.k, launchDTO.implementation, launchDTO.results);
		waitForKernel();
		time_lapsed = clock() - before;
		printf("Time for scanning: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

		copyArrayToHost(results, launchDTO.results, launchDTO.resultSize);

		printf("Done with scan \n");

		return results;
	}

	template <class T>
	void cleanupMW(LaunchDTO<T> launchDTO, int* dev_m_bounds, int* m_bounds, bool* dev_randomBitMap) {
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
	inline Point* runWeightedMinHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int implementation) {
		int numberOfThreads = calculateThreadsLocal(N_query);
		int numberOfBlocks = calculateBlocksLocal(N_query);
		int charSize = 255;
		clock_t before;
		clock_t time_lapsed;

		LaunchDTO<T> launchDTO = setupLaunchDTO<T>(implementation, DISTANCE_FUNCTION, k, d, sketchedDim, N_query, N_data, data, queries);
		printf("Done setting up DTO \n");

		//Setup query array.

		bool runOneBitMinHash = implementation != 4;

		int* dev_seedArr = createSeedArr(runOneBitMinHash ? sketchedDim * SKETCH_COMP_SIZE : sketchedDim);

		bool* dev_randomBitMap = generateRandomBoolVectors(charSize);

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
		setupMaps(dev_m_indexMap, dev_m_bounds, d, m_indexMapSize);


		printf("Starting sketch data \n");
		before = clock();

		sketchData(launchDTO, sketchedDim, launchDTO.sketchedData, launchDTO.sketchedQueries, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, runOneBitMinHash, SKETCH_COMP_SIZE, dev_randomBitMap, numberOfBlocks, numberOfThreads);

		//T* sketchedData = (T*)malloc(launchDTO.sketchedDataSize * sizeof(T)); 
		//copyArrayToHost(sketchedData, launchDTO.sketchedData, launchDTO.sketchedDataSize); 

		//printf("Printing \n");
		//for (int i = 0; i < launchDTO.N_data; i++) {
		//	printf("DataIdx: %d \n", i);
		//	for (int j = 0; j < launchDTO.sketchDim; j++)
		//		printf("%d ", sketchedData[i * launchDTO.sketchDim + j]);

		//	printf("\n");
		//}

		time_lapsed = clock() - before;
		printf("Time to hash on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

		//runBucketStatistics(launchDTO, numberOfThreads); 

		printf("Done sketching \nStarting scan \n");
		Point* results = runScan(launchDTO, numberOfBlocks, numberOfThreads);

		//Close
		cleanupMW(launchDTO, dev_m_bounds, m_bounds, dev_randomBitMap);

		resetDevice();

		return results;
	}

	template<class T, class K>
	void weightedMinHashGeneric(LaunchDTO<T> params, K* data, K* queries, int sketchedDim, int hashBits, bool runOneBitMinHash) {
		int numberOfThreads = calculateThreadsLocal(params.N_queries);
		int numberOfBlocks = calculateBlocksLocal(params.N_queries);
		int charSize = 255;
		clock_t before;
		clock_t time_lapsed;

		int* dev_seedArr = createSeedArr(runOneBitMinHash ? sketchedDim * hashBits : sketchedDim);

		bool* dev_randomBitMap = generateRandomBoolVectors(charSize);

		int* m_bounds = (int*)malloc(params.dimensions * sizeof(int));
		int* dev_m_bounds = mallocArray(m_bounds, params.dimensions);

		int* m_indexMapSizeArr = (int*)malloc(sizeof(int));
		int* dev_m_IndexMapSizeArr = mallocArray(m_indexMapSizeArr, 1);

		//Transform, Normalize, Maps
		minHashPreprocessing(params, dev_m_bounds, dev_m_IndexMapSizeArr, numberOfBlocks, numberOfThreads);

		int m_indexMapSize = getIndexMapSize(m_indexMapSizeArr, dev_m_IndexMapSizeArr);
		//	printf("Index map size: %d \n", m_indexMapSize);

			// Build maps
		int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
		int* dev_m_indexMap = createIndexMap(m_indexMapSize); //mallocArray(m_IndexMap, m_indexMapSize);

		//Finalize maps
		setupMaps(dev_m_indexMap, dev_m_bounds, params.dimensions, m_indexMapSize);


		printf("Starting sketch data \n");
		before = clock();

		sketchData(params, sketchedDim, data, queries, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, runOneBitMinHash, hashBits, dev_randomBitMap, numberOfBlocks, numberOfThreads);
	}

	Point* runMinHash(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int implementation) {

		if (implementation != 4) {
			return runWeightedMinHashLinearScan<unsigned int>(k, d, sketchedDim, N_query, N_data, data, queries, implementation);
		}
		else {
			return runWeightedMinHashLinearScan<unsigned char>(k, d, sketchedDim, N_query, N_data, data, queries, implementation);
		}
	}

}