
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

char* implementations[3] = { "OptimizedLinearScan", "MemOptimizedLinearScan", "SimHashLinearScan" };

int main(int argc, char **argv)
{
	//In arguments. 
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	int writeRes = atoi(argv[3]); //1 for yes, 0 for no.
	char* _k = argv[4];
	int implementation = atoi(argv[5]);
	int reportK = atoi(_k);
	int k = calculateK(reportK);
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

	printf("Implementation selected = %s\n", implementations[implementation-1]); 
	Point* res; 

	switch (implementation)
	{
	case 1:
		res = runOptimizedLinearScan(k, d, N_query, N_data, data, queries);
		break;
	case 2:
		res = runMemOptimizedLinearScan(k, d, N_query, N_data, data, queries);
		break;
	case 3: 
		res = runSimHashLinearScan(k, d, atoi(argv[6]), N_query, N_data, data, queries);
		break;
	default:
		printf("Invalid implementation selected. \n");
		//exit(-1);
		break; //?
	}

	if (writeRes) {
		writeResult(res, k, N_query, reportK);
	}

	printf("Starting to free \n"); 
	free(queries);
	free(data);
	printf("Success. Program exiting. \n");
	free(res);	
	//free(resDebug);
	return 0;

}

