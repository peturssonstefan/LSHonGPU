
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

const int MAX_THREADS = 1024; 

int calculateThreads(int querypoints) {
	if (querypoints < MAX_THREADS) return querypoints;
	else return MAX_THREADS;
}

int calculateBlocks(int querypoints) {
	if (querypoints < MAX_THREADS) return 1;
	else return ceil(querypoints / (float)MAX_THREADS); 
}

__global__
void add(int n_data,int n_query, int dimensions, int k, float *queryPoints, float *dataPoints, Point *results) {
	int queryIndex = blockIdx.x *blockDim.x + threadIdx.x;
	if (queryIndex < n_query) {

		float dotProduct;

		int index = queryIndex * dimensions;
		int result_index = queryIndex * k;
		//#pragma unroll; 
		for (int i = 0; i < n_data; i++) {
			float dotProduct = 0;
			float magnitude_query = 0.0;
			float magnitude_data = 0.0;
			for (int j = 0; j < dimensions; j++) {
				dotProduct += queryPoints[index + j] * dataPoints[dimensions*i + j];
				magnitude_query += queryPoints[index + j] * queryPoints[index + j];
				magnitude_data += dataPoints[dimensions*i + j] * dataPoints[dimensions*i + j];
			}

			magnitude_query = sqrt(magnitude_query);
			magnitude_data = sqrt(magnitude_data);
			float angular_distance = -(dotProduct / (magnitude_query * magnitude_data));

			Point currentPoint;
			currentPoint.ID = i;
			currentPoint.distance = angular_distance;
			Point swapPoint;

			for (int j = 0; (j < k && j <= i); j++) { // simple sorting.
				if (results[result_index + j].distance > currentPoint.distance) {
					swapPoint = results[result_index + j];
					results[result_index + j] = currentPoint;
					currentPoint = swapPoint;
				}
			}
		}
	}
}


int main(int argc, char **argv)
{
	//In arguments. 
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	char* _k = argv[3];
	int k = atoi(_k);

	//Data specific elements. Are read from datafiles. 
	int N_data = 0;
	int N_query = 0; 
	int d = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Is there a CUDA-capable GPU installed?");
		return -1;
	}
	float *x;
	float *y;
	Point *z;
	printf("Parsing files... \n");
	clock_t before = clock();
	x = parseFile(filepath_queries, N_query, d); 
	y = parseFile(filepath_data, N_data, d);
	clock_t time_lapsed = clock() - before; 

	printf("Time to read data files: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC)); 

	printf("Done parsing files. \n");
	printf("N_Query = %d \n", N_query);
	printf("N_Data = %d \n", N_data); 
	printf("k is set to: %d\n", k);
	z = (Point*)malloc(k*N_query * sizeof(Point));
	
	for (int i = 0; i < k * N_query; i++) {
		Point p;
		p.ID = -1;
		p.distance = 2.0f; //fill z array with default max value - given sim [-1,1]
		
		z[i] = p; 
	}

	float* dev_x = 0;
	float* dev_y = 0;
	Point* dev_z = 0;
	cudaMalloc((void**)&dev_x, N_query * d * sizeof(float));
	cudaMalloc((void**)&dev_y, N_data * d * sizeof(float));
	cudaMalloc((void**)&dev_z, k * N_query * sizeof(Point));

	cudaMemcpy(dev_x, x, N_query * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, N_data * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, z, k * N_query * sizeof(Point), cudaMemcpyHostToDevice);
	// initialize x and y arrays on the host

	int threads = calculateThreads(N_query); 
	int blocks = calculateBlocks(N_query); 
	printf("Threads: %d\n", threads); 
	printf("Blocks: %d\n", blocks);
	before = clock();
	add << <blocks, threads >> > (N_data, N_query, d, k, dev_x, dev_y, dev_z);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return -1;
	}

	time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	cudaStatus = cudaMemcpy(z, dev_z, k * N_query * sizeof(Point), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy from device to host returned error code %d \n", cudaStatus);
		return -1;
	}

	/*for (int i = 0; i < test; i++) {
		printf("%d:\n", i);
		for(int j = 0; j < k; j++)
			printf("%d %f\n", z[i*k + j].ID, z[i*k + j].distance); 
	}*/

	printf("Done. \n");

	//Free memory... 
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	free(x);
	free(y);
	free(z);

	cudaStatus = cudaDeviceReset();
	return 0;
}

