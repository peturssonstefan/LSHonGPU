
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<iostream>
#include "gloveparser.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "point.h"
#include <time.h>
#include "simpleLinearScan.cuh"
#include "optimizedLinearScan.cuh"
#include "simHash.cuh"
#include "resultWriter.h"
#include "memOptimizedLinearScan.cuh"
#include "launchHelper.cuh"
#include "validation.h"
#include "weightedMinHash.cuh"
#include "simHashJL.cuh"
#include "launchHelper.cuh"
#include "lshFramework.cuh"

char* implementations[6] = { "OptimizedLinearScan", "MemOptimizedLinearScan", "SimHashLinearScan", "WeightedMinHash", "OneBit - WeightedMinHash", "SimHash Johnson Lindenstrauss"};

Result linearScans(int implementation, int k, int d, int N_query, int N_data, float* data, float* queries, int sketchDim, int distanceFunc) {
	Result res;

	switch (implementation)
	{
	case 1:
		res = runOptimizedLinearScan(k, d, N_query, N_data, data, queries);
		break;
	case 2:
		res = runMemOptimizedLinearScan(k, d, N_query, N_data, data, queries, distanceFunc);
		break;
	case 3:
		res = simHash::runSimHashLinearScan(k, d, sketchDim, N_query, N_data, data, queries);
		break;
	case 4:
	case 5:
		res = weightedMinHash::runMinHash(k, d, sketchDim, N_query, N_data, data, queries, implementation);
		break;
	case 6:
		res = simHashJl::runSimHashJLLinearScan(k, d, sketchDim, N_query, N_data, data, queries);
		break;
	default:
		printf("Invalid implementation selected. \n");
		//exit(-1);
		break; //?
	}

	return res; 
}

template<class T, class K>
Result executeLSH(LaunchDTO<T> params, LshLaunchDTO<K> lshParams) {
	return runLsh(params, lshParams);
}

template<class T>
Result LshPipeline(LaunchDTO<T> params, int keysImplementation, int bucketKeyBits, int tables, bool runWithSketchedData) {
	
	switch (keysImplementation)
	{
	case 3:
		return executeLSH(params, setupLshLaunchDTO<unsigned short>(keysImplementation, bucketKeyBits, tables, params.N_data, params.N_queries, runWithSketchedData));
		break;
	case 4:
		return executeLSH(params, setupLshLaunchDTO<unsigned char>(keysImplementation, bucketKeyBits, tables, params.N_data, params.N_queries, runWithSketchedData));
		break;
	case 5:
		return executeLSH(params, setupLshLaunchDTO<unsigned short>(keysImplementation, bucketKeyBits, tables, params.N_data, params.N_queries, runWithSketchedData));
		break;
	case 6: 
		printf("Invalid implementation selected for LSH. \n");
		break; 
	default:
		printf("Invalid implementation selected for LSH. \n");
		//exit(-1);
		break; //?
	}

}



Result LSH(int implementation, int keysImplementation, int k, int d, int N_query, int N_data, float* data, float* queries, int sketchDim, int distanceFunc, int bucketKeyBits, int tables, bool runWithSketchedData) {

	switch (implementation)
	{
	case 3:
		return LshPipeline(setupLaunchDTO<unsigned int>(implementation, distanceFunc, k, d, sketchDim, N_query, N_data, data, queries), keysImplementation, bucketKeyBits, tables, runWithSketchedData);
		break;
	case 4:
		return LshPipeline(setupLaunchDTO<unsigned char>(implementation, distanceFunc, k, d, sketchDim, N_query, N_data, data, queries), keysImplementation, bucketKeyBits, tables, runWithSketchedData);
		break;
	case 5: 
		return LshPipeline(setupLaunchDTO<unsigned int>(implementation, distanceFunc, k, d, sketchDim, N_query, N_data, data, queries), keysImplementation, bucketKeyBits, tables, runWithSketchedData);
		break; 
	case 6:
		return LshPipeline(setupLaunchDTO<float>(implementation, distanceFunc, k, d, sketchDim, N_query, N_data, data, queries), keysImplementation, bucketKeyBits, tables, runWithSketchedData);
		break;
	default:
		printf("Invalid implementation selected for LSH. \n");
		//exit(-1);
		break; //?
	}

	return; 
}

int main(int argc, char **argv)
{
	//In arguments. 
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	char* filepath_truth = argv[3]; 
	int shouldRunValidation = atoi(argv[4]);
	int writeRes = atoi(argv[5]); //1 for yes, 0 for no.
	char* _k = argv[6];
	
	int implementation = atoi(argv[7]);
	int reportK = atoi(_k);
	int k = calculateK(reportK);
	int sketchDim = atoi(argv[8]);
	int distanceFunc = atoi(argv[9]); 
	int framework = atoi(argv[10]);
	int bucketKeyBits = atoi(argv[11]); 
	int tables = atoi(argv[12]); 
	int keysImplementation = atoi(argv[13]);
	bool runWithSketchedData = (bool)atoi(argv[14]);
	char* result_file_path = argv[15]; 
	int N_data = 0;
	int N_query = 0;
	int d = 0;
	float *queries;
	float *data;
	clock_t before = clock();
	queries = parseFile(filepath_queries, N_query, d);
	data = parseFile(filepath_data, N_data, d);
	clock_t time_lapsed = clock() - before;
	printf("Time to read data files: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));
	printf("Done parsing files. \n");
	printf("N_Query = %d \n", N_query);
	printf("N_Data = %d \n", N_data);
	printf("k is set to: %d\n", k);
	printf("Write res is set to %s \n", writeRes ? "True" : "False");
	printf("Validation is set to %s \n", shouldRunValidation ? "True" : "False");
	printf("Truth file is %s \n", filepath_truth);
	printf("Implementation selected = %s\n", implementations[implementation - 1]);

	//DTO's purely used for result writing. 
	LaunchDTO<int> defaultDTO = setupLaunchDTO<int>(implementation, distanceFunc, k, d, sketchDim, N_query, N_data, data, queries); 
	LshLaunchDTO<int> defaultLSHDTO = setupLshLaunchDTO<int>(keysImplementation, bucketKeyBits, tables, N_data, N_query, runWithSketchedData); 
	Result res; 

	if (framework == 0) {
		res = linearScans(implementation, k, d, N_query, N_data, data, queries, sketchDim, distanceFunc); 
	}
	else {
		res = LSH(implementation, keysImplementation, k, d, N_query, N_data, data, queries, sketchDim, distanceFunc, bucketKeyBits, tables, runWithSketchedData);
	}

	float* container = (float*)malloc(2 * sizeof(float)); //Dummy to avoid LINK ERR on cuda to cpp files

	if (shouldRunValidation) {
		printf("Running Validation: \n");
		runValidation(filepath_truth, container, res.results, N_query, k, reportK);
	}

	res.recall = container[0]; 
	res.avgDistance = container[1]; 

	if (writeRes) {
		printf("Writing results: \n");
		writeResult(res.results, k, N_query, reportK);
	}

	printf("From result dto: \n Preprocess Time: %d \n Construction Time: %d \n Scan Time: %d \n Recall: %f \n AvgDistance: %f \n", res.preprocessTime, res.constructionTime, res.scanTime, res.recall, res.avgDistance); 
	printf("Starting to free \n"); 
	free(queries);
	free(data);
	printf("Success. Program exiting. \n");
	free(res.results);	
	return 0;

}

