
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <direct.h>
#include <string>
#include <windows.h>
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
	char ch, filename[50], dim[5], currentdir[_MAX_PATH];
	int dimensions = 0;
	FILE *fp;

	if (GetCurrentDir(currentdir, sizeof(currentdir)) != NULL) {
		printf("Current working directory: %s\n", currentdir);
	}
	else {
		perror("getcwd() error");
		return 1;
	}

	//Get file name.
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

	float* matrix = (float *)malloc(glove_vector_count * dimensions * sizeof(float));

	printf("Created matrix \n"); 

	fp = fopen(fullpath, "r");
	if (fp == NULL) {
		perror("Error while opening file.");
	}

	printf("The content of file %s:\n", filename);
	bool number_is_negative = false; //Checks whether it is a negative number. 
	int number_counter = 0; //Number of numbers encountered. Perhaps necessary for later use... 
	int vector_counter = 0; //Number of vectors encountered. 
	int comma_counter = 0; //The placement of the next comma digit. 
	int index = 0; //The index in the matrix. 
	bool comma = false; //Whether a comma has been registered yet. 
	float x = 0; 
	while ((ch = fgetc(fp)) != EOF) {

		if (ch == 10) { //New line. 
			vector_counter++;
			number_counter = 0;
		}
		if (ch == 45) { //Negative number. 
			number_is_negative = true;
		}
		if (ch == 32) { //White space. 
			if (number_is_negative) {
				x = x * -1.0; 
			}

			matrix[index] = x; 
			number_counter++;
			number_is_negative = false; 
			comma = false; 
			comma_counter = 0; 
			x = 0; 
			index++;
		}
		if (isdigit(ch)) {
			if (!comma) { //No comma has been seen yet. 
				x = ch - 48; 
				continue; 
			}
			double digit = (ch - 48.0) / pow(10, comma_counter);
			x = x + digit; 
			comma_counter++; 
		}
		if (ch == 46) { //Comma. 
			comma = true;
			comma_counter++; 
		}
	}

	for (int i = 0; i < 100; i++) {
		int vectorId = i / 25; 
		printf("Number %.6f at location %d in matrix is connected to vector %d \n", matrix[i], i, vectorId); 
	}

	printf("Vectors counted: %d\n", vector_counter); 

	fclose(fp);
	free(matrix); 
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}

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
