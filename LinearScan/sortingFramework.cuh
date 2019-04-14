#pragma once

#include <cuda_runtime.h>
#include "point.h"
#include "constants.cuh"
#include "pointExtensions.cuh"
#include "sortParameters.h"
#include <math.h>

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
float broadCastMaxK(float maxK) {
	float otherVal = __shfl_sync(FULL_MASK, maxK, WARP_LEADER_THREAD);
	return maxK < otherVal ? otherVal : maxK;
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


__inline__ __device__
void setLane(Point& val, int lane, int otherLane, int otherId, float distance) {
	if (lane < otherLane) {
		if (!(val.distance > distance)) {
			val.ID = otherId;
			val.distance = distance; 
		}
	}
	else {
		if (!(val.distance < distance)) {
			val.ID = otherId;
			val.distance = distance;
		}
	}
}

__inline__ __device__
void subSort(Point& val,int size, int lane) {

	for (int offset = size / 2; offset > 0; offset /= 2) {
		
		int otherID = lane ^ offset; //__shfl_xor_sync(FULL_MASK, threadIdx.x, offset, WARPSIZE);
		int ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, WARPSIZE);
		float distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, WARPSIZE);
		
		bool direction = lane < otherID;
		//bool distanceDirection = valDistance > distance;

		val = direction ? max(val, createPoint(ID, distance)) : min(val, createPoint(ID, distance));

		/*int id = direction ? 
			distanceDirection ? valId : ID 
			: !distanceDirection ? valId : ID;

		float distanceVal = direction ?
			distanceDirection ? valDistance : distance
			: !distanceDirection ? valDistance : distance;

		
		valId = id;
		valDistance = distanceVal;*/


		//setLane(val, lane, otherID, ID, distance); 

	}
}

__inline__ __device__
void subSortUnrolled(Point& val, int lane) {

	for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {

		int otherID = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, WARPSIZE);//lane ^ offset;
		int ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, WARPSIZE);
		float distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, WARPSIZE);

		if (threadIdx.x < otherID) {
			/*val.ID = val.distance > distance ? val.ID : ID;
			val.distance = val.distance > distance ? val.distance : distance;*/

			val = max(val, createPoint(ID, distance));
		}
		else {
			/*val.ID = val.distance < distance ? val.ID : ID;
			val.distance = val.distance < distance ? val.distance : distance;*/

			val = min(val, createPoint(ID, distance));
		}
	}
}

//__inline__ __device__
//void printThreadQueue(Point* val) {
//	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
//		printf("T[%d] TQ[%d] = (%d, %f)\n", threadIdx.x, i, val[i].ID, val[i].distance);
//	}
//}

__inline__ __device__
void laneStrideSort(Point* val, Point swapPoint, Parameters& params) {


	// MEMORY ISSUE HERE - do not loop unroll 
	for (int pairSize = 1; pairSize <= WARPSIZE / 2; pairSize *= 2) {

		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			params.allIdx = params.lane + WARPSIZE * i;
			params.pairIdx = params.allIdx / pairSize;
			params.pairLane = params.allIdx % pairSize;
			params.exchangePairIdx = params.pairIdx % 2 == 0 ? params.pairIdx + 1 : params.pairIdx - 1;
			params.exchangeLane = (params.exchangePairIdx * pairSize + (pairSize - params.pairLane - 1)) % WARPSIZE;
			swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, params.exchangeLane, WARPSIZE);
			swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, params.exchangeLane, WARPSIZE);
			val[i] = params.lane < params.exchangeLane ? max(val[i], swapPoint) : min(val[i], swapPoint);
			subSort(val[i], pairSize * 2, params.lane); 
		}
	}


	for (int pairSize = WARPSIZE; pairSize <= (THREAD_QUEUE_SIZE * WARPSIZE) / 2; pairSize *= 2) {

		params.exchangeLane = (WARPSIZE - 1) - params.lane;
		params.elemsToExchange = pairSize / WARPSIZE * 2;

		for (int pairCouple = 0; pairCouple < ((params.allElemSize / pairSize) / 2); pairCouple++) {

			params.start = params.lane % 2 == 0 ? pairCouple * params.elemsToExchange : pairCouple * params.elemsToExchange + params.elemsToExchange - 1;
			params.increment = params.lane % 2 == 0 ? 1 : -1;
			params.end = params.elemsToExchange + (pairCouple * params.elemsToExchange);
			for (int i = params.start; i < params.end && i >= pairCouple * params.elemsToExchange; i += params.increment) {
				params.allIdx = params.lane + WARPSIZE * i;
				params.pairIdx = params.allIdx / pairSize;
				swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, params.exchangeLane, WARPSIZE);
				swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, params.exchangeLane, WARPSIZE);
				val[i] = params.pairIdx % 2 == 0 ? max(val[i], swapPoint) : min(val[i], swapPoint);
			}
			if (pairSize > WARPSIZE) {
				for (int i = pairCouple * params.elemsToExchange; i < pairCouple*params.elemsToExchange + params.elemsToExchange; i++) {
					for (int j = i; j < pairCouple*params.elemsToExchange + params.elemsToExchange; j++) {
						if (val[i].distance < val[j].distance) {
							swapPoint = val[i];
							val[i] = val[j];
							val[j] = swapPoint;
						}
					}
				}
			}
		}

		//#pragma unroll
		for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
			subSortUnrolled(val[i], params.lane);
		}
	}
}

__inline__ __device__ 
void simpleSort(Point* threadQueue, Point swapPoint) {
	for (int i = 0; i < THREAD_QUEUE_SIZE; i++) {
		for (int j = i; j < THREAD_QUEUE_SIZE; j++) {
			if (threadQueue[i].distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = threadQueue[i];
				threadQueue[i] = swapPoint;
			}
		}
	}
}

__inline__ __device__
void insertionSort(Point* threadQueue, Point swapPoint) {
	int i, j;
	for (i = 1; i < THREAD_QUEUE_SIZE; i++) {
		swapPoint = threadQueue[i];
		j = i - 1;

		while (j >= 0 && threadQueue[j].distance < swapPoint.distance) {
			threadQueue[j + 1] = threadQueue[j];
			j = j - 1;
		}
		threadQueue[j + 1] = swapPoint;
	}
}

__inline__ __device__
void startSort(Point* threadQueue, Point swapPoint, Parameters& params) {
	insertionSort(threadQueue, swapPoint);
	//simpleSort(threadQueue, swapPoint);
	laneStrideSort(threadQueue, swapPoint, params);
}
