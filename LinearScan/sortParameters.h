#include "constants.cuh"
#pragma once

struct Parameters {
	int lane = 0;
	int allElemSize = (THREAD_QUEUE_SIZE * WARPSIZE);
	int allIdx = 0;
	int pairIdx = 0;
	int pairLane = 0;
	int exchangePairIdx = 0;
	int exchangeLane = 0;
	int elemsToExchange = 0;
	int start = 0;
	int increment = 0;
	int end = 0;
};