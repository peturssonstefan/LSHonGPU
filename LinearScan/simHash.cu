#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"point.h"
#include"simHash.cuh"
#include <iostream>
#include <random>
#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


void checkError(cudaError_t cudaStatus, int line) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(cudaStatus), cudaStatus, line, __FILE__);
		throw "Error in optimizedLinearScan run.";
	}
}

__global__
void sketch(float* data, float* randomVectors, int size, int dimensions, int bits, int sketchDim, long* sketchedData) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int dataIndex = threadId * dimensions; 
	int sketchIndex = threadId * sketchDim; 
	int dataStride = blockDim.x*gridDim.x*dimensions; 
	int sketchStride = blockDim.x*gridDim.x*sketchDim;
	for (int i = dataIndex; i < size; i += dataStride) {
		long sketch = 0;
		int sketchCounter = 0;
		int dimStride = i + dimensions; 

		for (int j = sketchIndex; j < sketchIndex+sketchDim; j++) {
			long sketch = 0;
			int longSize = j + 1 * 64;
			for (int k = j*64; k < longSize ; k++) {
				float dotProduct = 0;
				for (int l = i; l < dimStride; l++) {
					dotProduct += data[i + l] * randomVectors[k];
				}
				if (dotProduct >= 0) {
					sketch |= 1 << (k%64);
				}
			}
			sketchedData[j] = sketch;
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

	std::normal_distribution<double> distribution(5.0, 2.0); // Mean,stddev

	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		std::cout << vectors[i] << std::endl;
	}

	return vectors;
}

Point* runSimHashLinearScan(int k, int d, int bits, int N_query, int N_data, float* data, float* queries) {

	int randomVectorsSize = d * bits; 
	int dataSize = d * N_data; 
	int querySize = d * N_query;
	int sketchedDim = bits / 64; 
	int sketchedDataSize = N_data * sketchedDim; 
	int sketchedQuerySize = N_query * sketchedDim;
	float* randomVectors = generateRandomVectors(randomVectorsSize);
	
	//Setup random vector array.
	float* dev_randomVectors = 0; 
	checkError(cudaMalloc((void**)&dev_randomVectors, randomVectorsSize * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(dev_randomVectors, randomVectors, randomVectorsSize * sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	
	//Setup data array.
	float* dev_data = 0;
	checkError(cudaMalloc((void**)&dev_data, dataSize * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(dev_data, data, dataSize * sizeof(float), cudaMemcpyHostToDevice), __LINE__);

	//Setup query array.
	float* dev_queries = 0;
	checkError(cudaMalloc((void**)&dev_queries, querySize * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(dev_queries, queries, querySize * sizeof(float), cudaMemcpyHostToDevice), __LINE__);

	//Setup sketchedData array.
	long* sketchedData = (long*)malloc(sketchedDataSize * sizeof(long));
	long* dev_sketchedData = 0;
	checkError(cudaMalloc((void**)&dev_sketchedData, sketchedDataSize * sizeof(long)), __LINE__);
	checkError(cudaMemcpy(dev_sketchedData, sketchedData, sketchedDataSize * sizeof(long), cudaMemcpyHostToDevice), __LINE__);

	//Setup sketchedQuery array.
	long* sketchedQuery = (long*)malloc(sketchedQuerySize * sizeof(long)); 
	long* dev_sketchedQuery = 0;
	checkError(cudaMalloc((void**)&dev_sketchedQuery, sketchedQuerySize * sizeof(long)), __LINE__);
	checkError(cudaMemcpy(dev_sketchedQuery, sketchedQuery, sketchedQuerySize * sizeof(long), cudaMemcpyHostToDevice), __LINE__);

	sketch << <1,1>> > (dev_data, dev_randomVectors, dataSize, d, bits, sketchedDim, dev_sketchedData);

	checkError(cudaDeviceSynchronize(), __LINE__);
	checkError(cudaDeviceReset(), __LINE__);
	return nullptr;
}