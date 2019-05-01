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
		int ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, warpSize);
		float distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, warpSize);
		
		bool direction = lane < otherID;

		val = direction ? max(val, createPoint(ID, distance)) : min(val, createPoint(ID, distance));

	 }
}

__inline__ __device__
void subSortUnrolled(Point& val, int lane) {

	for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {

		int otherID = __shfl_xor_sync(FULL_MASK, threadIdx.x, offset, WARPSIZE);//lane ^ offset;
		int ID = __shfl_xor_sync(FULL_MASK, val.ID, offset, WARPSIZE);
		float distance = __shfl_xor_sync(FULL_MASK, val.distance, offset, WARPSIZE);

		if (threadIdx.x < otherID) {
			val = max(val, createPoint(ID, distance));
		}
		else {
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


	int otherID; 
	int ID = 0; 
	float distance; 
	bool direction;
	int threadQueueSize = params.allElemSize / warpSize; 
	// MEMORY ISSUE HERE - do not loop unroll 
	for (int pairSize = 1; pairSize <= warpSize / 2; pairSize *= 2) {

		for (int i = 0; i < threadQueueSize; i++) {
			params.allIdx = params.lane + warpSize * i;
			params.pairIdx = params.allIdx / pairSize;
			params.pairLane = params.allIdx % pairSize;
			params.exchangePairIdx = params.pairIdx % 2 == 0 ? params.pairIdx + 1 : params.pairIdx - 1;
			params.exchangeLane = (params.exchangePairIdx * pairSize + (pairSize - params.pairLane - 1)) % warpSize;
			swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, params.exchangeLane, warpSize);
			swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, params.exchangeLane, warpSize);
			val[i] = params.lane < params.exchangeLane ? max(val[i], swapPoint) : min(val[i], swapPoint);
			subSort(val[i], pairSize * 2, params.lane); 
		}
	}

	int maxPairSize = (threadQueueSize * warpSize) / 2; 

	for (int pairSize = WARPSIZE; pairSize <= maxPairSize; pairSize *= 2) {

		params.exchangeLane = (warpSize - 1) - params.lane;
		params.elemsToExchange = pairSize / warpSize * 2;

		for (int pairCouple = 0; pairCouple < ((params.allElemSize / pairSize) / 2); pairCouple++) {

			params.start = params.lane % 2 == 0 ? pairCouple * params.elemsToExchange : pairCouple * params.elemsToExchange + params.elemsToExchange - 1;
			params.increment = params.lane % 2 == 0 ? 1 : -1;
			params.end = params.elemsToExchange + (pairCouple * params.elemsToExchange);
			for (int i = params.start; i < params.end && i >= pairCouple * params.elemsToExchange; i += params.increment) {
				params.allIdx = params.lane + warpSize * i;
				params.pairIdx = params.allIdx / pairSize;
				swapPoint.ID = __shfl_sync(FULL_MASK, val[i].ID, params.exchangeLane, warpSize);
				swapPoint.distance = __shfl_sync(FULL_MASK, val[i].distance, params.exchangeLane, warpSize);
				val[i] = params.pairIdx % 2 == 0 ? max(val[i], swapPoint) : min(val[i], swapPoint);
			}
			if (pairSize > warpSize) {
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

		for (int i = 0; i < threadQueueSize; i++) {
			subSort(val[i], warpSize, params.lane);
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


__inline__ __device__
void threadQueueSort(Point* threadQueue, Point currentPoint, Point swapPoint, int maxKDistance, int& queuePosition, int candidateSetSize, Parameters params) {
	if (WITH_TQ_OR_BUFFER) {
		//run TQ
		for (int j = candidateSetSize - 1; j >= 0; j--) { // simple sorting.
			if (currentPoint.distance < threadQueue[j].distance) {
				swapPoint = threadQueue[j];
				threadQueue[j] = currentPoint;
				currentPoint = swapPoint;
			}
		}


		//Verify that head of thread queue is not smaller than biggest k distance.
		if (__ballot_sync(FULL_MASK, threadQueue[0].distance < maxKDistance) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
		}
	}
	else {
		//run buffer
		if (currentPoint.distance < maxKDistance || same(currentPoint, maxKDistance)) {
			threadQueue[queuePosition++] = currentPoint;
		}



		if (__ballot_sync(FULL_MASK, queuePosition >= candidateSetSize) && __activemask() == FULL_MASK) {
			startSort(threadQueue, swapPoint, params);
			maxKDistance = broadCastMaxK(threadQueue[candidateSetSize].distance);
			//printQueue(threadQueue);
			queuePosition = 0;
		}
	}
}
