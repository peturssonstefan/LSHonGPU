#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"point.h"
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
#include "randomVectorGenerator.h"
#include "processingUtils.cuh"

namespace simHashJl {

	template<class T>__global__
		void normalizeVectors(LaunchDTO<T> launchDTO) {
		transformToUnitVectors(launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions);
		transformToUnitVectors(launchDTO.data, launchDTO.N_data, launchDTO.dimensions);
	}

	template<class T>__global__
		void sketch(LaunchDTO<T> launchDTO, float* data, float* randomVectors, int dataSize, T* sketchedData) {
		int threadId = blockDim.x * blockIdx.x + threadIdx.x;
		int numberOfThreads = blockDim.x * gridDim.x;

		for (int i = threadId; i < dataSize; i += numberOfThreads) {
			int pointIndex = i * launchDTO.dimensions;
			int sketchIndex = i * launchDTO.sketchDim;
			for (int sketchBlockId = 0; sketchBlockId < launchDTO.sketchDim; sketchBlockId++) {
				float dot = 0;
				for (int dimIndex = 0; dimIndex < launchDTO.dimensions; dimIndex++) {
					dot += data[pointIndex + dimIndex] * randomVectors[launchDTO.dimensions * sketchBlockId + dimIndex];
				}

				sketchedData[sketchIndex + sketchBlockId] = dot;
			}
		}
	}

	template<class T>
	void runGenericJL(LaunchDTO<T> params) {

		int numberOfThreads = calculateThreadsLocal(params.N_queries);
		int numberOfBlocks = calculateBlocksLocal(params.N_queries);
		int randomVectorSize = params.sketchDim * params.dimensions;
		normalizeVectors << <numberOfBlocks, numberOfThreads >> > (params);
		waitForKernel();
		float* randomVectors = generateRandomVectors(randomVectorSize, params.sketchDim);
		float* dev_randomVectors = mallocArray(randomVectors, randomVectorSize, true);

		sketch << <numberOfBlocks, numberOfThreads >> > (params, params.data, dev_randomVectors, params.N_data, params.sketchedData);
		waitForKernel();
		sketch << <numberOfBlocks, numberOfThreads >> > (params, params.queries, dev_randomVectors, params.N_queries, params.sketchedQueries);
		waitForKernel();

		freeDeviceArray(dev_randomVectors);
		free(randomVectors);
	}

	__global__
		void scan(LaunchDTO<float> launchDTO) {

		int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
		int queryIndex = warpId * launchDTO.dimensions;
		if (queryIndex < launchDTO.dimensions * launchDTO.N_queries) {
			scanHammingDistanceJL(launchDTO);
		}
	}

	void cleanup(LaunchDTO<float> launchDTO, float* randomVectors, float* dev_randomVectors) {
		freeDeviceArray(launchDTO.data);
		freeDeviceArray(launchDTO.queries);
		freeDeviceArray(launchDTO.sketchedData);
		freeDeviceArray(launchDTO.sketchedQueries);
		freeDeviceArray(dev_randomVectors);
		freeDeviceArray(launchDTO.results);
		free(randomVectors);
	}

	Point* runSimHashJLLinearScan(int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries) {

		setDevice();
		int numberOfThreads = calculateThreadsLocal(N_query);
		int numberOfBlocks = calculateBlocksLocal(N_query);
		int randomVectorSize = sketchedDim * d;
		LaunchDTO<float> launchDTO = setupLaunchDTO<float>(6, 3, k, d, sketchedDim, N_query, N_data, data, queries);
		simHashJl::normalizeVectors << <numberOfBlocks, numberOfThreads >> > (launchDTO);
		waitForKernel();
		float* randomVectors = generateRandomVectors(randomVectorSize, sketchedDim);
		float* dev_randomVectors = mallocArray(randomVectors, randomVectorSize, true);

		sketch << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.data, dev_randomVectors, N_data, launchDTO.sketchedData);
		waitForKernel();
		sketch << <numberOfBlocks, numberOfThreads >> > (launchDTO, launchDTO.queries, dev_randomVectors, N_query, launchDTO.sketchedQueries);
		waitForKernel();

		float* sketchedData = (float*)malloc(launchDTO.sketchedDataSize * sizeof(float));
		copyArrayToHost(sketchedData, launchDTO.sketchedData, launchDTO.sketchedDataSize);


		scan << <numberOfBlocks, numberOfThreads >> > (launchDTO);
		waitForKernel();
		Point* results = (Point*)malloc(launchDTO.resultSize * sizeof(Point));
		copyArrayToHost(results, launchDTO.results, launchDTO.resultSize);

		cleanup(launchDTO, randomVectors, dev_randomVectors);
		resetDevice();

		return results;
	}

}