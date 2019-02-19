
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

int main(int argc, char **argv)
{
	//In arguments. 
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	char* _k = argv[3];
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

	// TODO args for selecting implementation
	//Point* resDebug = runSimpleLinearScan(k, d, N_query, N_data, data, queries);
	//printf("Done with simple scan \n \n"); 
	Point* res = runOptimizedLinearScan(k, d, N_query, N_data, data, queries);


	free(queries);
	free(data);
	free(res);	
	//free(resDebug);
}

