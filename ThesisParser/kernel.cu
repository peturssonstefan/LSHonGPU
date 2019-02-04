
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

int getDimensions(); 
char* getDataFile();

const int glove_vector_count = 1193514;



__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(){

	char* filepath = getDataFile();
	int dimensions = getDimensions();
	float* matrix = parseFile(filepath, dimensions);

	//Print 100 first items. 
	for (int i = 0; i < 100; i++) {
		printf("%f \n", matrix[i]);
	}

	free(filepath);
	free(matrix);
	
	return 0;
}

//Get the data file needed. 
char* getDataFile() {
	char filename[50], currentdir[_MAX_PATH];
	if (GetCurrentDir(currentdir, sizeof(currentdir)) != NULL) {
		printf("Current working directory: %s\n", currentdir);
	}
	else {
		perror("getcwd() error");
		exit(1); 
	}
	printf("Enter filename: \n");
	fgets(filename, sizeof(filename), stdin);
	filename[strlen(filename) - 1] = '\0';
	char* fullpath = (char*)malloc(_MAX_PATH * sizeof(char));
	strcpy(fullpath, currentdir);
	strcat(fullpath, "\\datasets\\");
	strcat(fullpath, filename);
	return fullpath; 
}

//Get the number of dimensions. 
int getDimensions() {
	int dimensions = 0;
	char dim[5];
	printf("Enter number of dimensions: \n");
	fgets(dim, sizeof(dim), stdin);
	dimensions = atoi(dim);
	return dimensions; 
}