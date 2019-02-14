
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<iostream>
#include "gloveparser.cuh"
#include <stdio.h>

__global__
void add(int n, int d, float *x, float *y, float *z) {
	 

	float dotProduct; 
	int queryIndex = threadIdx.x;
	int index = queryIndex * d;

	for (int i = 0; i < n; i++) {
		float dotProduct = 0; 
		float magnitude_x = 0.0;
		float magnitude_y = 0.0;
		for (int j = 0; j < d; j++) {
			dotProduct += x[index + j] * y[d*i + j];
			magnitude_x += x[index + j] * x[index + j];
			magnitude_y += y[d*i + j] * y[d*i + j];
		}

		magnitude_x = sqrt(magnitude_x);
		magnitude_y = sqrt(magnitude_y);
		z[queryIndex * n + i] = -(dotProduct / (magnitude_x * magnitude_y)); 
		if (i == 0) {
			printf("Q %d = %f \n",queryIndex, z[queryIndex*n]); 
		}
	}
}


int main(int argc, char **argv)
{
	char* filepath_data = argv[1];
	char* filepath_queries = argv[2];
	//char* dim = argv[3];
	//char* _k = argv[3];
	//int dimensions = atoi(dim);
	//int k = atoi(_k);

	int N_data = 0;
	int N_query = 0; 
	int d = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}
	float *x;
	float *y;
	float *z;
	printf("Parsing files... \n");
	x = parseFile(filepath_queries, N_query, d); 
	y = parseFile(filepath_data, N_data, d);
	printf("Done parsing files. \n");
	z = (float*)malloc(N_data*N_query * sizeof(float)); 

	/*for (int i = 0; i < N_query * d; i++) {
		x[i] = 1.0f;
	}

	for (int i = 0; i < N_data*d; i++) {
		y[i] = 2.0f;
	}*/

	float* dev_x = 0;
	float* dev_y = 0;
	float* dev_z = 0;
	cudaMalloc((void**)&dev_x, N_query * d * sizeof(float));
	cudaMalloc((void**)&dev_y, N_data * d * sizeof(float));
	cudaMalloc((void**)&dev_z, N_data * N_query * sizeof(float));

	cudaMemcpy(dev_x, x, N_query * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, N_data * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z, y, N_data * N_query * sizeof(float), cudaMemcpyHostToDevice);
	// initialize x and y arrays on the host

	add << <1, N_query>> > (N_data, d, dev_x, dev_y, dev_z);

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

	cudaStatus = cudaMemcpy(z, dev_z, N_data * N_query * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return -1;
	}

	//for (int i = 0; i < N_query*d; i++) { 
	//	printf
	//}

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	free(x);
	free(y);

	cudaStatus = cudaDeviceReset();
	return 0;
}

