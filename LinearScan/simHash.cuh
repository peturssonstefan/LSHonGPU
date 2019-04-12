#pragma once
#include "point.h"
#include <cuda.h>
#include <bitset>
#include <math.h>
#include <time.h>
#include "constants.cuh"
#include "sketchedDistanceScanners.cuh"
#include "launchHelper.cuh"
#include "cudaHelpers.cuh"
#include "randomVectorGenerator.h"


Point* runSimHashLinearScan(int k, int d, int bits, int N_query, int N_data, float* data, float* queries);

template <class T> __global__
void sketchDataGeneric(float* data, float* randomVectors, int size, int dimensions, int sketchDim, int bucketKeyBits, T* sketchedData) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int numberOfThreads = blockDim.x * gridDim.x;
	for (int i = threadId; i < size; i += numberOfThreads) {
		int pointIndex = i * dimensions;
		int sketchIndex = i * sketchDim;
		for (int sketchBlockId = 0; sketchBlockId < sketchDim; sketchBlockId++) {
			unsigned int sketch = 0;
			for (int bitIndex = 0; bitIndex < bucketKeyBits; bitIndex++) {
				float dot = 0;
				int randomVectorIndex = bucketKeyBits * dimensions * sketchBlockId + bitIndex * dimensions;
				for (int dimIndex = 0; dimIndex < dimensions; dimIndex++) {
					dot += data[pointIndex + dimIndex] * randomVectors[randomVectorIndex + dimIndex];
				}
				unsigned int bit = dot >= 0 ? 1 : 0;
				sketch |= bit << bitIndex;

			}

			sketchedData[sketchIndex + sketchBlockId] = sketch;
		}
	}
}