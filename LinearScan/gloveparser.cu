#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "gloveparser.cuh"

float* parseFile(char* path, int& n, int& dimensions) {
	FILE *fp;
	float* list;
	fp = fopen(path, "r");
	if (fp == NULL) {
		printf("Error while opening file: %s. \n", path);
		exit(-1);
	}

	bool number_is_negative = false; //Checks whether it is a negative number. 
	int ch = 0;
	int comma_counter = 0; //The placement of the next comma digit. 
	int index = 0; //The index in the matrix. 
	bool comma = false; //Whether a comma has been registered yet. 
	bool isID = true;
	float x = 0;
	printf("Parsing file.\n");

	char _n[256], d[256];
	fgets(_n, sizeof(_n), fp);
	fgets(d, sizeof(d), fp);

	n = atoi(_n);
	dimensions = atoi(d);

	list = (float*)malloc(n * dimensions * sizeof(float));

	while ((ch = fgetc(fp)) != EOF) {

		while (ch != 10 && ch != EOF) {
			if (ch == 32) {
				if (isID) { //Just continue after this. Otherwise we will be adding an unnecassary 0. 
					isID = false;
				}
				else { //Add number. 
					if (number_is_negative) x = x * -1.0; //Check if negative. 
					//printf("Adding to list[%d] value %f \n", index, x);
					list[index] = x;
					//Reset values.
					x = 0;
					index++;
					comma = false;
					number_is_negative = false;
					comma_counter = 0;
				}

			}
			else if (isID) { // Is the number the ID. 
				//Id is omitted for now.
			}
			else if (isdigit(ch)) { //Is it a number. 
				if (comma) { //If comma, than compute correct digit. 
					double digit = (ch - 48.0) / pow(10, comma_counter);
					x = x + digit;
					comma_counter++;
				}
				else { //Otherwise just assign number. 
					x = ch - 48.0;
				}
			}
			else if (ch == 45) { //Set negative flag. 
				number_is_negative = true;
			}
			else if (ch == 46) { //Set comma flag. 
				comma = true;
				comma_counter++;
			}
			ch = fgetc(fp); //Get next character. 
		}

		x = 0;
		comma = false;
		comma_counter = 0;
		isID = true;
	}

	printf("Done in parsing loop");

	fclose(fp);
	return list;
}