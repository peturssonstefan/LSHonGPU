#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

__inline__ __device__ 
void bucketDistribution(unsigned char* hashes, int hashesSize, int* res) {
	int threadId = (blockDim.x * blockIdx.x) + threadIdx.x; 
	for (int i = threadIdx.x; i < hashesSize; i += gridDim.x * blockDim.x) {
		atomicAdd(&res[hashes[i]],1); 
	}
}

