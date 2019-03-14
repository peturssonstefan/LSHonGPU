#pragma once
#include <cuda_runtime.h>
#include "constants.cuh"
#include "sortParameters.h"
#include<math.h>

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
	int divisor = k / WARPSIZE;
	return divisor * WARPSIZE + WARPSIZE; 
}