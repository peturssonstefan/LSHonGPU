
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <direct.h>
#include <string>
#include <windows.h>
#include "gloveparser.cuh"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#define GetCurrentDir _getcwd

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
const int glove_vector_count = 1193514;

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	char filename[50], dim[5], currentdir[_MAX_PATH];
	int dimensions = 0;


	if (GetCurrentDir(currentdir, sizeof(currentdir)) != NULL) {
		printf("Current working directory: %s\n", currentdir);
	}
	else {
		perror("getcwd() error");
		return 1;
	}

	//Get the file name for the data set.
	printf("Enter filename: \n");
	fgets(filename, sizeof(filename), stdin);
	filename[strlen(filename) - 1] = '\0';
	char fullpath[_MAX_PATH];
	strcpy_s(fullpath, currentdir);
	strcat(fullpath, "\\datasets\\");
	strcat(fullpath, filename);
	printf("Full path is: %s\n", fullpath);

	//Get dimensions.
	printf("Enter number of dimensions: \n");
	fgets(dim, sizeof(dim), stdin);
	dimensions = atoi(dim);
	printf("Registered %d dimensions \n", dimensions);

	float* matrix = parseFile(fullpath, dimensions);

	for (int i = 0; i < 100; i++) {
		printf("%f", matrix[i]);
	}

	free(matrix);
	
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
