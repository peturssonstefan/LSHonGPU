#pragma once 
#include "point.h"
#include "pointExtensions.cuh"
#include <cuda.h>
#include <bitset>
#include <math.h>
#include <time.h>
#include "constants.cuh"
#include "sketchedDistanceScanners.cuh"
#include "launchHelper.cuh"
#include "cudaHelpers.cuh"
#include "randomVectorGenerator.h"

namespace crosspoly {
	
	template<class T>__global__
		void normalizeVectors(LaunchDTO<T> launchDTO) {
		transformToUnitVectors(launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions);
		transformToUnitVectors(launchDTO.data, launchDTO.N_data, launchDTO.dimensions);
	}

	template <class T, class K>
	void sketchCrossPolyTope(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, int numberOfBlocks, int numberOfThreads) {
		int threadId = blockDim.x * blockIdx.x + threadIdx.x;
		int totalThreads = blockDim.x * gridDim.x;
		normalizeVectors << <numberOfBlocks, numberOfThreads >> > (params);

	}


}
