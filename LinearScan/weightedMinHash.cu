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


#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\


#define DISTANCE_FUNCTION 2

__global__
void preprocess(float* data, float* queries, int N_data, int N_queries, int dimensions,int* m_bounds, int* m_indexMapSize) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	
	transformData(data, queries, N_data, N_queries, dimensions, m_bounds); 

	__syncthreads();

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
float uniformRandom(curandState* state, int seed) {
	float val = curand_uniform(state); 
	return val; 
}

__inline__ __device__ 
bool isGreen(int* m_indexMap, int* m_bounds, float* data, float r, int i, int d) {
	int rIdx = r + 1;
	int componentIdx = m_indexMap[rIdx];
	int m_bounds_val = m_bounds[componentIdx];
	float pointDI = data[i*d + componentIdx];
	return r <= m_bounds_val + pointDI; 

}


__global__
void sketchData(float* data, int N_data, int dimensions, int sketchDim, int* m_indexMap, int* m_bounds, int M, int* seeds, unsigned char* sketchedData) {
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
					float random = uniformRandom(&s, seed);
					r = M * random;
					red = !isGreen(m_indexMap, m_bounds, data, r, i, dimensions);
					if (red) {
						char val = sketchedData[i * sketchDim + hashIdx];
						counter++;
					}
				}

				if (counter > 0) {
					//if (threadId == 0) printf("counter: %d, bitindex: %d, seed: %d, r: %d  \n", counter, bitIndex, seed, r);
					sketchedData[i * sketchDim + hashIdx] |= 1 << bitIndex;
				}

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
void scan(float* originalData, float* originalQueries, int dimensions, unsigned char * data, unsigned char * queries, int sketchDim, int N_data, int N_query, int k, Point* result) {
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
	int queryIndex = warpId * dimensions;
	if (queryIndex < dimensions * N_query) {
		scanHammingDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, DISTANCE_FUNCTION,result);
	}
}

Point* runWeightedMinHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {
	int componentSize = 8;

	int numberOfThreads = calculateThreadsLocal(N_query);
	int numberOfBlocks = calculateBlocksLocal(N_query);
	int bits = sketchedDim * componentSize;
	
	int dataSize = d * N_data;
	int querySize = d * N_query;
	int resultSize = k * N_query;

	int m_indexMapSize = 0;

	//Setup data array.
	float* dev_data = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_data, dataSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_data, data, dataSize * sizeof(float), cudaMemcpyHostToDevice));
	
	//Setup query array.
	float* dev_queries = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_queries, querySize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_queries, queries, querySize * sizeof(float), cudaMemcpyHostToDevice));
	
	//Seeds
	int seedArrSize = sketchedDim * componentSize;
	int* seedArr = (int*)malloc(seedArrSize * sizeof(int));
	int* dev_seedArr;

	for (int i = 0; i < seedArrSize; i++) {
		seedArr[i] = i * 1234 + 92138;
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_seedArr, seedArrSize * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_seedArr, seedArr, seedArrSize * sizeof(int), cudaMemcpyHostToDevice));

	//Sketch arrays
	int sketchedDataSize = N_data * sketchedDim;
	unsigned char* sketchedData = (unsigned char*)malloc(sketchedDataSize * sizeof(unsigned char));
	unsigned char* dev_sketchedData;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_sketchedData, sketchedDataSize * sizeof(unsigned char)));

	int sketchedQueriesSize = N_query * sketchedDim;
	unsigned char* sketchedQueries = (unsigned char*)malloc(sketchedQueriesSize * sizeof(unsigned char));
	unsigned char* dev_sketchedQueries;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_sketchedQueries, sketchedQueriesSize * sizeof(unsigned char)));

	int* m_bounds = (int*)malloc(d * sizeof(int));
	int* dev_m_bounds;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_m_bounds, d * sizeof(int)));

	int* m_indexMapSizeArr = (int*)malloc(sizeof(int));
	int* dev_m_IndexMapSizeArr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_m_IndexMapSizeArr, sizeof(int)));

	// Transform data
	preprocess << <1, numberOfThreads >> > (dev_data, dev_queries, N_data, N_query, d, dev_m_bounds, dev_m_IndexMapSizeArr);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(m_indexMapSizeArr, dev_m_IndexMapSizeArr,sizeof(int), cudaMemcpyDeviceToHost));
	m_indexMapSize = m_indexMapSizeArr[0];
	printf("Index map size: %d \n", m_indexMapSize);

	// Build maps
	int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
	int* dev_m_indexMap;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_m_indexMap, m_indexMapSize * sizeof(int)));
	setupMapIndex << <1, 1 >> > (dev_m_bounds, dev_m_indexMap, d, m_indexMapSize); //
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaMemcpy(m_IndexMap, dev_m_indexMap, m_indexMapSize * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(m_bounds, dev_m_bounds, d * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < d; i++) {
		printf("%d ", m_bounds[i]);
	}
	printf("\n");
	for (int i = 0; i < m_indexMapSize; i++) {
		printf("%d ", m_IndexMap[i]);
	}
	printf("\n");

	printf("Starting sketch data \n");
	sketchData << <numberOfBlocks, numberOfThreads >> > (dev_queries, N_query, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr ,dev_sketchedQueries);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	sketchData << <numberOfBlocks, numberOfThreads >> > (dev_data, N_data, d, sketchedDim, dev_m_indexMap, dev_m_bounds, m_indexMapSize, dev_seedArr, dev_sketchedData);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	// Do linear scan
	Point* results = (Point*)malloc(resultSize * sizeof(Point));
	Point* dev_results;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_results, resultSize * sizeof(Point)));
	printf("Done sketching \n Starting scan \n");
	scan << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_queries, d, dev_sketchedData, dev_sketchedQueries, sketchedDim, N_data, N_query, k, dev_results);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaMemcpy(results, dev_results, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

	printf("Done with scan \n");
	//Close
	CUDA_CHECK_RETURN(cudaFree(dev_data));
	CUDA_CHECK_RETURN(cudaFree(dev_queries));
	CUDA_CHECK_RETURN(cudaFree(dev_sketchedData));
	CUDA_CHECK_RETURN(cudaFree(dev_sketchedQueries));
	CUDA_CHECK_RETURN(cudaFree(dev_m_bounds));
	CUDA_CHECK_RETURN(cudaFree(dev_results));
	CUDA_CHECK_RETURN(cudaDeviceReset());



	free(sketchedData);
	free(sketchedQueries);
	free(m_bounds);
	return results;
}