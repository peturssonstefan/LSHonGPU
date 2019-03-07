
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

char* implementations[3] = { "OptimizedLinearScan", "MemOptimizedLinearScan", "SimHashLinearScan" };

int main(int argc, char **argv)
{
	//In arguments. 
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	char* _k = argv[3];
	int implementation = atoi(argv[4]);
	int k = atoi(_k);
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
		res = runSimHashLinearScan(k, d, atoi(argv[5]), N_query, N_data, data, queries);
		break;
	default:
		printf("Invalid implementation selected. \n");
		//exit(-1);
		break; //?
	}

	//writeResult(res, k, N_query); 
	//writeOnlyIDs(res, k, N_query); 

	printf("Starting to free \n"); 
	free(queries);
	free(data);
	printf("Success. Program exiting. \n");
	free(res);	
	//free(resDebug);
	return 0;

}

