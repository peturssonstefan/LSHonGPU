#pragma once

#include <cuda_runtime.h>
#include "point.h"
#include "constants.cuh"

template <typename T>
inline __device__ T* shuffle_down(T* const val, unsigned int delta, int width = warpSize) {
	//static assert(sizeof(T*) == sizeof(long long), "pointer size incorrect"); 
	long long v = (long long)val;
#if CUDA_VERSION >= 9000
	return (T*)__shfl_down_sync(FULL_MASK, v, delta);
#else
	return (T*)__shfl_down(v, delta);
#endif
}


__inline__ __device__
Point* warpReduceArrays(Point* val, int k) {
	Point merged[MAX_K];
	for (int offset = 16; offset > 0; offset /= 2) {
		Point* tmpVal = shuffle_down(val, offset);
		int i = 0;
		int j = 0;

		for (int x = 0; x < k; x++) {
			if (val[i].distance <= tmpVal[j].distance) {
				merged[x] = val[i++];
			}
			else if (val[i].distance > tmpVal[j].distance) {
				merged[x] = tmpVal[j++];
			}
		}

		for (int i = 0; i < k; i++) {
			val[i] = merged[i];
		}
	}

	return val;
}

__inline__ __device__
Point* blockReduce(Point* val, int k, float maxValue) {
	__shared__ Point* shared[32];

	int lane = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;
	val = warpReduceArrays(val, k);

	if (lane == 0) {
		shared[warpId] = val;
	}

	__syncthreads();

	//static __shared__ Point maxArray[MAX_K];
	static Point maxArray[MAX_K];
#pragma unroll
	for (int i = 0; i < MAX_K; i++) {
		Point p;
		p.ID = -1;
		p.distance = maxValue;
		maxArray[i] = p;
	}

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : maxArray;

	if (warpId == 0) {
		val = warpReduceArrays(val, k);
	}

	return val;
}