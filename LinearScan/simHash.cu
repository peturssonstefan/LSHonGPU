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
#include "cudaHelpers.cuh"
#include "randomVectorGenerator.h"

namespace simHash {

#define DISTANCE_FUNCTION 1
#define IMPLEMENTATION 3

	__global__
		void sketch(float* data, float* randomVectors, int size, int dimensions, int sketchDim, unsigned int* sketchedData) {
		int threadId = blockDim.x * blockIdx.x + threadIdx.x;
		int numberOfThreads = blockDim.x * gridDim.x;

		for (int i = threadId; i < size; i += numberOfThreads) {
			int pointIndex = i * dimensions;
			int sketchIndex = i * sketchDim;
			for (int sketchBlockId = 0; sketchBlockId < sketchDim; sketchBlockId++) {
				unsigned int sketch = 0;
				for (int bitIndex = 0; bitIndex < SKETCH_COMP_SIZE; bitIndex++) {
					float dot = 0;
					int randomVectorIndex = SKETCH_COMP_SIZE * dimensions * sketchBlockId + bitIndex * dimensions;
					for (int dimIndex = 0; dimIndex < dimensions; dimIndex++) {
						dot += data[pointIndex + dimIndex] * randomVectors[randomVectorIndex + dimIndex];
					}
					unsigned int bit = dot >= 0 ? 1 : 0;
					sketch |= bit << bitIndex;

				}

				sketchedData[sketchIndex + sketchBlockId] = sketch;
			}
		}

	}

	__global__
		void scan(float* originalData, float* originalQueries, int dimensions, unsigned int * data, unsigned int * queries, int sketchDim, int N_data, int N_query, int k, Point* result) {

		int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
		int queryIndex = warpId * dimensions;
		if (warpId < N_query) {
			scanHammingDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, DISTANCE_FUNCTION, IMPLEMENTATION, result);
		}
	}



	Result runSimHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {

		setDevice();
		int numberOfThreads = calculateThreadsLocal(N_query);
		int numberOfBlocks = calculateBlocksLocal(N_query);
		int bits = sketchedDim * SKETCH_COMP_SIZE;
		int randomVectorsSize = d * bits;
		int dataSize = d * N_data;
		int querySize = d * N_query;
		int sketchedDataSize = N_data * sketchedDim;
		int sketchedQuerySize = N_query * sketchedDim;
		int resultSize = N_query * k;
		float* randomVectors = generateRandomVectors(randomVectorsSize);
		Result res;
		res.setupResult(N_query, k);
		//Setup random vector array.
		float* dev_randomVectors = mallocArray(randomVectors, randomVectorsSize, true);

		//Setup data array.
		float* dev_data = mallocArray(data, dataSize, true);

		//Setup query array.
		float* dev_queries = mallocArray(queries, querySize, true);

		//Setup sketchedData array.
		unsigned int* sketchedData = (unsigned int*)malloc(sketchedDataSize * sizeof(unsigned int));
		unsigned int* dev_sketchedData = mallocArray(sketchedData, sketchedDataSize);

		//Setup sketchedQuery array.
		unsigned int* sketchedQuery = (unsigned int*)malloc(sketchedQuerySize * sizeof(unsigned int));
		unsigned int* dev_sketchedQuery = mallocArray(sketchedQuery, sketchedQuerySize);

		clock_t before = clock();
		printf("Started hashing \n");
		sketch << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_randomVectors, N_data, d, sketchedDim, dev_sketchedData);
		waitForKernel();

		sketch << <numberOfBlocks, numberOfThreads >> > (dev_queries, dev_randomVectors, N_query, d, sketchedDim, dev_sketchedQuery);
		waitForKernel();

		printf("Done sketching.\n");
		clock_t time_lapsed = clock() - before;
		res.constructionTime = (time_lapsed * 1000 / CLOCKS_PER_SEC);
		printf("Time to hash on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

		//copyArrayToHost(sketchedData, dev_sketchedData, sketchedDataSize);
		//copyArrayToHost(sketchedQuery, dev_sketchedQuery, sketchedQuerySize);

		//Setup Result Array 
		Point* results = (Point*)malloc(resultSize * sizeof(Point));
		Point* dev_results = mallocArray(results, resultSize);

		printf("Calculating Distance. \n");
		before = clock();
		scan << <N_query, numberOfThreads >> > (dev_data, dev_queries, d, dev_sketchedData, dev_sketchedQuery, sketchedDim, N_data, N_query, k, dev_results);
		waitForKernel();

		time_lapsed = clock() - before;
		printf("Time to calculate distance on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
		res.scanTime = (time_lapsed * 1000 / CLOCKS_PER_SEC);
		copyArrayToHost(results, dev_results, resultSize);
		res.copyResultPoints(results, N_query, k); 

		//Close
		freeDeviceArray(dev_data);
		freeDeviceArray(dev_queries);
		freeDeviceArray(dev_sketchedData);
		freeDeviceArray(dev_sketchedQuery);
		freeDeviceArray(dev_results);
		free(sketchedData);
		free(sketchedQuery);
		free(results); 
		resetDevice();

		return res;
	}

}