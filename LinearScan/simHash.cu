#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"point.h"
#include"simHash.cuh"
#include <iostream>
#include <random>
#include <cuda.h>
#include <bitset>
#include <math.h>

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

void checkError(cudaError_t cudaStatus, int line) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(cudaStatus), cudaStatus, line, __FILE__);
		throw "Error in optimizedLinearScan run.";
	}
}

__global__
void sketch(float* data, float* randomVectors, int size, int dimensions, int bits, int sketchDim, unsigned long* sketchedData) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int numberOfThreads = blockDim.x * gridDim.x;

	for (int i = threadId; i < size; i += numberOfThreads) {
		int pointIndex = i * dimensions;
		int sketchIndex = i * sketchDim;
		for (int sketchBlockId = 0; sketchBlockId < sketchDim; sketchBlockId++) {
			long sketch = 0;
			for (int bitIndex = 0; bitIndex < bits; bitIndex++) {
				float dot = 0;
				int randomVectorIndex = bits * dimensions * sketchBlockId + bitIndex * dimensions; 
				for (int dimIndex = 0; dimIndex < dimensions; dimIndex++) {
					//printf("Tid[%d]: D - R: %d - %d\n", threadId, pointIndex + dimIndex, randomVectorIndex + dimIndex);
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
		std::cout << vectors[i] << ",";
	}
	std::cout << std::endl; 
	return vectors;
}

Point* runSimHashLinearScan(int k, int d, int bits, int N_query, int N_data, float* data, float* queries) {

	int randomVectorsSize = d * bits; 
	int dataSize = d * N_data; 
	int querySize = d * N_query;
	int sketchedDim = std::ceil(bits / 64.0f); 
	printf("sketchDim: %d \n", sketchedDim);
	int sketchedDataSize = N_data * sketchedDim; 
	int sketchedQuerySize = N_query * sketchedDim;
	float* randomVectors = generateRandomVectors(randomVectorsSize);
	
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

	sketch << <1,2>> > (dev_data, dev_randomVectors, dataSize, d, bits, sketchedDim, dev_sketchedData);
	sketch << <1,2>> > (dev_queries, dev_randomVectors, N_query, d, bits, sketchedDim, dev_sketchedQuery);


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(sketchedData, dev_sketchedData, sketchedDataSize * sizeof(long), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(sketchedQuery, dev_sketchedQuery, sketchedQuerySize * sizeof(unsigned long), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceReset());


	for (int i = 0; i < 10; i++) {
		/*std::bitset<64> bitset(sketchedQuery[i]);

		std::cout << bitset << std::endl; */
		for (int j = 0; j < 64; j++) {

			unsigned long bit = (sketchedQuery[i] >> j) & 1UL; 
			printf("%d", bit); 
		}
		printf(" --- %lu    query\n", sketchedQuery[i]);
		for (int j = 0; j < 64; j++) {

			unsigned long bit = (sketchedData[i] >> j) & 1UL;
			printf("%d", bit);
		}
		printf(" --- %lu    data\n", sketchedData[i]);
	}



	return nullptr;
}