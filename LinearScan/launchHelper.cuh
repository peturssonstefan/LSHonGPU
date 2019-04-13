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
LaunchDTO<T> setupLaunchDTO(int implementation, int distanceFunc, int k, int d, int sketchedDim, int N_queries, int N_data, float* data, float* queries) {
	LaunchDTO<T> launchDTO;

	launchDTO.implementation = implementation;
	launchDTO.distanceFunc = distanceFunc; 
	launchDTO.k = k;
	launchDTO.dimensions = d;
	launchDTO.sketchDim = sketchedDim;
	launchDTO.N_data = N_data;
	launchDTO.N_queries = N_queries;
	launchDTO.dataSize = N_data * d;
	launchDTO.querySize = N_queries * d;
	launchDTO.resultSize = N_queries * k;
	launchDTO.sketchedDataSize = N_data * sketchedDim;
	launchDTO.sketchedQueriesSize = N_queries * sketchedDim;
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

template <class T>
LshLaunchDTO<T> setupLshLaunchDTO(int keyImplementation, int bucketKeyBits, int tables, int N_data, int N_queries) {
	T* dataKeys; 
	T* queryKeys;

	LshLaunchDTO<T> lshLaunchDTO; 
	lshLaunchDTO.bucketKeyBits = bucketKeyBits;
	lshLaunchDTO.tables = tables;
	lshLaunchDTO.keyImplementation = keyImplementation;
	lshLaunchDTO.tableSize = 1 << bucketKeyBits;
	lshLaunchDTO.dataKeys = mallocArray(dataKeys, tables * N_data);
	lshLaunchDTO.queryKeys = mallocArray(queryKeys, tables * N_queries); 
	return lshLaunchDTO;
}