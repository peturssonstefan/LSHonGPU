#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK_RETURN(value){ \
	cudaError_t _m_cudaStat = value; \
	if (_m_cudaStat != cudaSuccess) { \
		fprintf(stderr, "Error: %s \n Error Code %d \n Error is at line %d, in file %s \n", cudaGetErrorString(_m_cudaStat), _m_cudaStat, __LINE__ , __FILE__); \
		cudaDeviceReset(); \
		exit(-1); \
	} \
}\

template <class T>
T* mallocArray(T* host_array, int arraySize, bool copyData = false) {
	T* dev_array;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_array, arraySize * sizeof(T)));

	if (copyData)
		CUDA_CHECK_RETURN(cudaMemcpy(dev_array, host_array, arraySize * sizeof(T), cudaMemcpyHostToDevice));

	return dev_array;
}

template<class T>
void copyArrayToHost(T* destination, T* source, int size) {
	CUDA_CHECK_RETURN(cudaMemcpy(destination, source, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T>
void freeDeviceArray(T* dev_array) {
	CUDA_CHECK_RETURN(cudaFree(dev_array));
}

void waitForKernel();

void resetDevice();

void setDevice(int device = 0);


