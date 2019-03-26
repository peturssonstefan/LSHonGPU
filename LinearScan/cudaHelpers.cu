#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "cudaHelpers.cuh"

void waitForKernel() {
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void resetDevice() {
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void setDevice(int device) {
	CUDA_CHECK_RETURN(cudaSetDevice(device));
}