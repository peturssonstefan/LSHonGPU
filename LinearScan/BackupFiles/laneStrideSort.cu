
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <iostream>
#include <random>
#include <limits>
#include <algorithm>

#define THREAD_QUEUE_SIZE 8
#define WARPQUEUE_SIZE 1
#define FULL_MASK 0xffffffff
#define WARP_SIZE 8

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		exit(-1); \
	} \
}\

template <typename T>
inline __device__ T* shuffle_xor(T* const val, int laneMask, int width = warpSize) {
	long long v = (long long)val;
	return (T*)__shfl_xor_sync(FULL_MASK, v, laneMask, width);
}

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


struct Point {
	int ID;
	float distance;
};

__inline__ __device__
Point min(Point p1, Point p2) {
	return p1.distance < p2.distance ? p1 : p2;
}

__inline__ __device__
Point max(Point p1, Point p2) {
	return p1.distance > p2.distance ? p1 : p2;
}

__inline__ __device__
Point createPoint(int ID, float distance) {
	Point p;
	p.ID = ID;
	p.distance = distance;
	return p;
}

__inline__ __device__
void printIntArr(int* val, int size) {
	for (int i = 0; i < size; i++) printf("T[%d]: arr[%d] = %d \n", threadIdx.x, i, val[i]);
}

__inline__ __device__
int subSort(int val, int size) {

	for (int offset = size / 2; offset > 0; offset /= 2) {

		int otherTid = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, size);
		int otherVal = __shfl_xor_sync(FULL_MASK, val, offset, size);
		val = threadIdx.x < otherTid ? max(val, otherVal) : min(val, otherVal);
	}

	return val;

}

__inline__ __device__
int* laneStrideSort(int* val) {
	int tid = threadIdx.x,
		int lane = tid % WARP_SIZE;
	int allElemSize = (THREAD_QUEUE_SIZE * WARP_SIZE);
	int largestPairSize = allElemSize / 2;

	for (int pairSize = 1; pairSize <= WARP_SIZE / 2; pairSize *= 2) {

		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			int allIdx = lane + WARP_SIZE * i;
			int pairIdx = allIdx / pairSize;
			int pairLane = allIdx % pairSize;
			int exchangePairIdx = pairIdx % 2 == 0 ? pairIdx + 1 : pairIdx - 1;
			int exchangeLane = (exchangePairIdx * pairSize + (pairSize - pairLane - 1)) % WARP_SIZE;
			int otherVal = __shfl_sync(FULL_MASK, val[i], exchangeLane, WARP_SIZE);
			//printf("(I,P): (%d,%d)  -  T[%d] to T[%d] vals: (%d,%d) \n ", i, pairSize, tid, exchangeLane, val[i], otherVal);
			val[i] = lane < exchangeLane ? max(val[i], otherVal) : min(val[i], otherVal);

			val[i] = subSort(val[i], pairSize * 2);
		}
	}

	for (int pairSize = WARP_SIZE; pairSize <= largestPairSize; pairSize *= 2) {

		int exchangeLane = (WARP_SIZE - 1) - lane;
		int pairCoupleSize = (allElemSize / pairSize) / 2;
		int elemsToExchange = pairSize / WARP_SIZE * 2;

		for (int pairCouple = 0; pairCouple < pairCoupleSize; pairCouple++) {

			int start = lane % 2 == 0 ? pairCouple * elemsToExchange : pairCouple * elemsToExchange + elemsToExchange - 1;
			int increment = lane % 2 == 0 ? 1 : -1;
			int end = elemsToExchange + (pairCouple * elemsToExchange);
			for (int i = start; i < end && i >= pairCouple * elemsToExchange; i += increment) {
				int allIdx = lane + WARP_SIZE * i;
				int pairId = allIdx / pairSize;
				int otherVal = __shfl_sync(FULL_MASK, val[i], exchangeLane, WARP_SIZE);
				val[i] = pairId % 2 == 0 ? max(val[i], otherVal) : min(val[i], otherVal);
				printf("(Size,Couple): (%d,%d)  -  T[%d] to T[%d] vals: (%d,%d) \n ", pairSize, pairCouple, tid, exchangeLane, val[i], otherVal);
			}
			if (pairSize > WARP_SIZE) {
				for (int i = pairCouple * elemsToExchange; i < pairCouple*elemsToExchange + elemsToExchange; i++) {
					for (int j = i; j < pairCouple*elemsToExchange + elemsToExchange; j++) {
						if (val[i] < val[j]) {
							int tmp = val[i];
							val[i] = val[j];
							val[j] = tmp;
						}
					}
				}
			}
		}

		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			val[i] = subSort(val[i], WARP_SIZE);
		}

	}

	return val;

}


__global__
void kernel() {
	int start = 8;
	int tid = threadIdx.x % (WARP_SIZE / 2);
	int offset1 = 2 * tid;
	int offset2 = offset1 + 1;
	int arr[THREAD_QUEUE_SIZE];
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		int localOff = i % 2 == 0 ? offset1 : offset2; 
		arr[i] = start - localOff; 
		printf("T[%d]: arr[%d] = %d\n", tid, i, arr[i]);
	}
	
	
	int* res = laneStrideSort(arr);
	printIntArr(res, THREAD_QUEUE_SIZE);
}

int main()
{

	kernel << <1, WARP_SIZE >> > ();

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
