#pragma once
#include <cuda_runtime.h>
#include "constants.cuh"
#include "sortParameters.h"
#include<math.h>
#include "launchDTO.h"

__inline__ __host__ __device__ 
int calculateThreadsLocal(int querypoints) {
	if (querypoints * WARPSIZE < MAX_THREADS) return querypoints * WARPSIZE;
	else return MAX_THREADS;
}

__inline__ __host__ __device__
int calculateBlocksLocal(int querypoints) {
	if (querypoints * WARPSIZE < MAX_THREADS) return 1;
	else return ceil(querypoints / (float)WARPSIZE);
}

__inline__ __host__ __device__
int calculateK(int k) {
	int divisor = (k-1) / WARPSIZE;
	return divisor * WARPSIZE + WARPSIZE; 
}

template <class T>
LaunchDTO<T> setupLaunchDTO(int implementation, int distanceFunc, int k, int d, int sketchedDim, int N_query, int N_data, float* data, float* queries, int bucketKeyBits, int tables) {
	LaunchDTO<T> launchDTO;

	launchDTO.implementation = implementation;
	launchDTO.distanceFunc = distanceFunc; 
	launchDTO.k = k;
	launchDTO.dimensions = d;
	launchDTO.bucketKeyBits = bucketKeyBits;
	launchDTO.tables = tables; 
	launchDTO.sketchDim = sketchedDim;
	launchDTO.N_data = N_data;
	launchDTO.N_queries = N_query;
	launchDTO.dataSize = N_data * d;
	launchDTO.querySize = N_query * d;
	launchDTO.resultSize = N_query * k;
	launchDTO.sketchedDataSize = N_data * sketchedDim;
	launchDTO.sketchedQueriesSize = N_query * sketchedDim;
	Point* results = (Point*)malloc(launchDTO.resultSize * sizeof(Point));
	T* sketchedData;
	T* sketchedQueries;

	launchDTO.data = mallocArray(data, launchDTO.dataSize, true);
	launchDTO.queries = mallocArray(queries, launchDTO.querySize, true);
	launchDTO.results = mallocArray(results, launchDTO.resultSize);
	launchDTO.sketchedData = mallocArray(sketchedData, launchDTO.sketchedDataSize);
	launchDTO.sketchedQueries = mallocArray(sketchedQueries, launchDTO.sketchedQueriesSize);
	free(results);
	return launchDTO;
}
