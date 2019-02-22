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
void add(int n_data, int n_query, int dimensions, int k, float *queryPoints, float *dataPoints, Point *results) {
	int queryIndex = blockIdx.x *blockDim.x + threadIdx.x;
	if (queryIndex < n_query) {

		float dotProduct;

		int index = queryIndex * dimensions;
		int result_index = queryIndex * k;
		//#pragma unroll; 
		for (int i = 0; i < n_data; i++) {
			float dotProduct = 0;
			/*for (int j = 0; j < dimensions; j++) {
				dotProduct += queryPoints[index + j] * dataPoints[dimensions*i + j];
			}*/

			float angular_distance = -(i);

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



Point* runSimpleLinearScan(int k, int d, int N_query, int N_data, float* data, float* queries) {
	//Data specific elements. Are read from datafiles. 

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Is there a CUDA-capable GPU installed?");
		throw "Error in simpleLinearScan run."; 
	}

	Point *z;
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

	cudaMemcpy(dev_x, queries, N_query * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, data, N_data * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, z, k * N_query * sizeof(Point), cudaMemcpyHostToDevice);
	// initialize x and y arrays on the host

	int threads = calculateThreads(N_query);
	int blocks = calculateBlocks(N_query);
	printf("Threads: %d\n", threads);
	printf("Blocks: %d\n", blocks);
	clock_t before = clock();
	add << <blocks, threads >> > (N_data, N_query, d, k, dev_x, dev_y, dev_z);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		throw "Error in simpleLinearScan run.";
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		throw "Error in simpleLinearScan run.";
	}

	clock_t time_lapsed = clock() - before;
	printf("Time calculate on the GPU: %d \n", (time_lapsed * 1000 / CLOCKS_PER_SEC));

	cudaStatus = cudaMemcpy(z, dev_z, k * N_query * sizeof(Point), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda memcpy from device to host returned error code %d \n", cudaStatus);
		throw "Error in simpleLinearScan run.";
	}

	for (int i = 0; i < 4; i++) {
		printf("Query: %d\n", i); 
		for (int j = 0; j < k; j++) {
			printf("ID: %d dist: %f\n", z[j + i*k].ID, z[j +i*k].distance);
		}
	}

	printf("Done. \n");

	//Free memory... 
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);

	cudaStatus = cudaDeviceReset();
	return z;
}