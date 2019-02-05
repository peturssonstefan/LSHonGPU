#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <direct.h>
#include <string>
#include <windows.h>

const int glove_vector_count = 1193514;

float* parseFile(char* path, int dimensions) {
	FILE *fp;
	printf("Path: %s \n", path); 
	printf("Dimensions: %d \n", dimensions);
	float* matrix = (float *)malloc(glove_vector_count * dimensions * sizeof(float));
	printf("Created matrix \n");
	fp = fopen(path, "r");
	if (fp == NULL) {
		perror("Error while opening file.");
	}

	bool number_is_negative = false; //Checks whether it is a negative number. 
	int ch = 0;
	int number_counter = 0; //Number of numbers encountered. Perhaps necessary for later use... 
	int vector_counter = 0; //Number of vectors encountered. 
	int comma_counter = 0; //The placement of the next comma digit. 
	int index = 0; //The index in the matrix. 
	bool comma = false; //Whether a comma has been registered yet. 
	float x = 0;

	printf("Parsing file.\n");

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

	fclose(fp);

	printf("Done parsing.\n");

	return matrix;
}