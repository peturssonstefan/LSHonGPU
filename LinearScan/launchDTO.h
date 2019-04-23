#pragma once

#include "point.h"

template <class T>
struct LaunchDTO
{
	int implementation;
	int distanceFunc; 
	float* data;
	float* queries;
	int N_data;
	int N_queries; 
	int dimensions;
	T* sketchedData; 
	T* sketchedQueries; 
	int sketchedDataSize;
	int sketchedQueriesSize; 
	int sketchDim; 
	int k; 
	int dataSize; 
	int querySize; 
	int resultSize; 
	Point* results; 
};


template <class T>
struct LshLaunchDTO
{
	T* dataKeys; 
	T* queryKeys;
	int keyImplementation; 
	int bucketKeyBits;
	int tables;
	int tableSize; 
	bool runWithSketchedData; 
};