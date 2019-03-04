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

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

__global__
void sketch(float* data, float* randomVectors, int size, int dimensions, int sketchDim, unsigned long* sketchedData) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int numberOfThreads = blockDim.x * gridDim.x;

	for (int i = threadId; i < size; i += numberOfThreads) {
		int pointIndex = i * dimensions;
		int sketchIndex = i * sketchDim;
		for (int sketchBlockId = 0; sketchBlockId < sketchDim; sketchBlockId++) {
			long sketch = 0;
			for (int bitIndex = 0; bitIndex < SKETCH_COMP_SIZE; bitIndex++) {
				float dot = 0;
				int randomVectorIndex = SKETCH_COMP_SIZE * dimensions * sketchBlockId + bitIndex * dimensions;
				for (int dimIndex = 0; dimIndex < dimensions; dimIndex++) {
					dot += data[pointIndex + dimIndex] * randomVectors[randomVectorIndex + dimIndex];
				}
				unsigned long bit = dot >= 0 ? 1 : 0;
				sketch |= bit << bitIndex;
				
			}

			sketchedData[sketchIndex + sketchBlockId] = sketch;
		}
	}

}

float* generateRandomVectors(int N, bool randomSeed = false) {

	// same seed 
	static float* vectors = (float*)malloc(N * sizeof(float));
	std::default_random_engine generator;
	// different seeds
	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator

	std::normal_distribution<double> distribution(0.0, 1.0); // Standard normal distribution.

	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		//std::cout << vectors[i] << ",";
	}
	//std::cout << std::endl; 
	return vectors;
}

__global__
void scan(unsigned long * data, unsigned long * queries, int sketchDim, int N_data, int N_query, int k, Point* threadQueue, Point* result) {
	int threadQueueIndex = (blockDim.x * blockIdx.x + threadIdx.x) * k; 
	scanHammingDistance(data, queries, sketchDim, N_data, N_query, k, &threadQueue[threadQueueIndex], result); 
}

 

Point* runSimHashLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {

	int numberOfThreads = 1024; 
	int numberOfBlocks = 3;
	int bits = sketchedDim * SKETCH_COMP_SIZE;
	int randomVectorsSize = d * bits; 
	int dataSize = d * N_data; 
	int querySize = d * N_query;
	int sketchedDataSize = N_data * sketchedDim; 
	int sketchedQuerySize = N_query * sketchedDim;
	int threadQueueSize = N_query * numberOfThreads * k; 
	int resultSize = N_query * k; 
	float* randomVectors = generateRandomVectors(randomVectorsSize);
	CUDA_CHECK_RETURN(cudaSetDevice(0));
	//Setup random vector array.
	float* dev_randomVectors = 0; 
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_randomVectors, randomVectorsSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_randomVectors, randomVectors, randomVectorsSize * sizeof(float), cudaMemcpyHostToDevice));
	
	//Setup data array.
	float* dev_data = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_data, dataSize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_data, data, dataSize * sizeof(float), cudaMemcpyHostToDevice));

	//Setup query array.
	float* dev_queries = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_queries, querySize * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_queries, queries, querySize * sizeof(float), cudaMemcpyHostToDevice));

	//Setup sketchedData array.
	unsigned long* sketchedData = (unsigned long*)malloc(sketchedDataSize * sizeof(unsigned long));
	unsigned long* dev_sketchedData = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_sketchedData, sketchedDataSize * sizeof(unsigned long)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_sketchedData, sketchedData, sketchedDataSize * sizeof(unsigned long), cudaMemcpyHostToDevice));

	//Setup sketchedQuery array.
	unsigned long* sketchedQuery = (unsigned long*)malloc(sketchedQuerySize * sizeof(unsigned long));
	unsigned long* dev_sketchedQuery = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_sketchedQuery, sketchedQuerySize * sizeof(unsigned long)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_sketchedQuery, sketchedQuery, sketchedQuerySize * sizeof(unsigned long), cudaMemcpyHostToDevice));
	clock_t before = clock();
	printf("Started hashing \n");
	sketch << <numberOfBlocks, numberOfThreads >> > (dev_data, dev_randomVectors, N_data, d, sketchedDim, dev_sketchedData);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	printf("Done with first sketch \n");
	sketch << <numberOfBlocks, numberOfThreads >> > (dev_queries, dev_randomVectors, N_query, d, sketchedDim, dev_sketchedQuery);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	printf("Done sketching queries.\n");
	clock_t time_lapsed = clock() - before;
	printf("Time to hash on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	CUDA_CHECK_RETURN(cudaMemcpy(sketchedData, dev_sketchedData, sketchedDataSize * sizeof(long), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(sketchedQuery, dev_sketchedQuery, sketchedQuerySize * sizeof(unsigned long), cudaMemcpyDeviceToHost));

	//Setup Thread Queue Array 
	Point* threadQueue = (Point*)malloc(threadQueueSize * sizeof(Point));
	Point* dev_threadQueue = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_threadQueue, threadQueueSize * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_threadQueue, threadQueue, threadQueueSize * sizeof(Point), cudaMemcpyHostToDevice));

	//Setup Result Array 
	Point* results = (Point*)malloc(resultSize * sizeof(Point));
	Point* dev_results = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_results, resultSize * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_results, results, resultSize * sizeof(Point), cudaMemcpyHostToDevice));
	printf("Calculating Distance. \n");
	before = clock();
	scan << <N_query, numberOfThreads >> > (dev_sketchedData, dev_sketchedQuery, sketchedDim, N_data, N_query, k, dev_threadQueue, dev_results);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	time_lapsed = clock() - before;
	printf("Time to calculate distance on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	CUDA_CHECK_RETURN(cudaMemcpy(results, dev_results, resultSize * sizeof(Point), cudaMemcpyDeviceToHost));

	//Close
	CUDA_CHECK_RETURN(cudaFree(dev_sketchedData));
	CUDA_CHECK_RETURN(cudaFree(dev_sketchedQuery));
	CUDA_CHECK_RETURN(cudaFree(dev_threadQueue));
	CUDA_CHECK_RETURN(cudaFree(dev_results));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return results;
}