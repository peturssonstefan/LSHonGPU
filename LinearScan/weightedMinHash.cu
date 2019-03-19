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


#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\


__global__
void transformData(float* data, float* queries, int N_data, int N_queries, int dimensions,int* m_bounds, int* m_indexMapSize) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	
	// Find min
	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMin(&m_bounds[dim], (int)data[i * dimensions + dim]); // floor by casting
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			atomicMin(&m_bounds[dim], (int)queries[i * dimensions + dim]); // floor by casting
		}
	}

	__syncthreads();

	// Transform data
	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			data[i * dimensions + dim] += abs(m_bounds[dim]);
		}
	}

	for (int i = threadId; i < N_queries; i += totalThreads) {
		for (int dim = 0; dim < dimensions; dim++) {
			queries[i * dimensions + dim] += abs(m_bounds[dim]);
		}
	}

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
}

__global__
void setupMapIndex(int* m_bounds, int* indexToComponentMap, int dimensions, int indexMapSize) {
	if (threadIdx.x == 0) { //TODO... We all know there is a smarter way to do this...
		int currentBound = 0;
		for (int i = 0; i < dimensions; i++) {
			int bound = currentBound + m_bounds[i];
			for (int j = currentBound; j < bound; j++) {
				indexToComponentMap[j] = i;
			}

			m_bounds[i] = currentBound;
			currentBound = bound;
		}
	}
}

__inline__ __device__
float uniformRandom(curandState* state, int seed) {
	curand_init(seed,0, 0, &state[0]); 
	float val = curand_uniform(state); 
	return val; 
}

__inline__ __device__ 
bool isGreen(int* m_indexMap, int* m_bounds, float* data, int r, int i, int d) {
	int componentIdx = m_indexMap[r];
	int m_bounds_val = m_bounds[componentIdx];
	float pointDI = data[i*d + componentIdx];

	/*if (threadIdx.x == 0) {
		printf("Recieved r: %d \n", r);
		printf("componentIdx is: %d \n", componentIdx);
		printf("m_bounds_val: %d \n", m_bounds_val);
		printf("pointDI: %f \n", pointDI);
		printf("isRed = %d\n \n", r <= m_bounds_val + pointDI);
	}*/
	return r <= m_bounds_val + pointDI; 

}

__global__
void sketchData(float* data, int N_data, int dimensions, int sketchDim ,int* m_indexMap, int* m_bounds, int M, int* seeds, unsigned char* sketchedData) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	curandState s; 

	for (int i = threadId; i < N_data; i += totalThreads) {
		for (int hashIdx = 0; hashIdx < sketchDim; hashIdx++) {
			int seed = seeds[hashIdx];
			bool red = true;

			while (red) {
				float random = uniformRandom(&s, seed);
				int r = M * random;
				red = !isGreen(m_indexMap, m_bounds, data, r, i, dimensions);
				seed = random * 10000;
				if (red) {
					char val = sketchedData[i * sketchDim + hashIdx]; 
					if (val == 255) printf("val is 255 for thread %d \n", threadId); 
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
		scanHammingDistance(originalData, &originalQueries[queryIndex], dimensions, data, queries, sketchDim, N_data, N_query, k, result);
	}
}

Point* runWeightedMinHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {
	int componentSize = sizeof(unsigned char);

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
	int* seedArr = (int*)malloc(sketchedDim * sizeof(int));
	int* dev_seedArr;

	for (int i = 0; i < sketchedDim; i++) {
		seedArr[i] = i * 1234; 
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_seedArr, sketchedDim * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_seedArr, seedArr, sketchedDim * sizeof(int), cudaMemcpyHostToDevice));

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
	transformData << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_queries, N_data, N_query, d, dev_m_bounds, dev_m_IndexMapSizeArr);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(m_indexMapSizeArr, dev_m_IndexMapSizeArr,sizeof(int), cudaMemcpyDeviceToHost));
	m_indexMapSize = m_indexMapSizeArr[0];
	printf("Index map size: %d \n", m_indexMapSize);

	// Build maps
	int* m_IndexMap = (int*)malloc(m_indexMapSize * sizeof(int));
	int* dev_m_indexMap;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_m_indexMap, m_indexMapSize * sizeof(int)));
	setupMapIndex << <numberOfBlocks, numberOfThreads >> > (dev_m_bounds, dev_m_indexMap, d, m_indexMapSize);
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

	printf("Starting linear scan\n");

	scan << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_queries, d, dev_sketchedData, dev_sketchedQueries, sketchedDim, N_data, N_query, k, dev_results);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	printf("Done scan\n");

	CUDA_CHECK_RETURN(cudaMemcpy(results, dev_results, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

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
	printf("Got here\n");
	return results;
}