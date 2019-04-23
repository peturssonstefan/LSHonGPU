#pragma once
#include "point.h"
#include "resultDTO.h"
#include "constants.cuh"
#include "launchDTO.h"

void writeResult(Point* results, int k, int n_query, int reportK);

void writeOnlyIDs(Point* results, int k, int n_query);

void writePerformanceResults(Result result, LaunchDTO<int> launchDTO, LshLaunchDTO<int> lshLaunchDTO, char* resultFile);