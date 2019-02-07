#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "gloveparser.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const int glove_vector_count = 1193514;

thrust::host_vector<Point> parseFile(char* path, int dimensions) {
	FILE *fp;
	thrust::host_vector<Point> points;
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

	while ((ch = fgetc(fp)) != EOF) {
		Point p;	
		p.ID = 0;
		while (ch != 10 && ch != EOF) {
			if (ch == 32) {
				if (isID) { //Just continue after this. Otherwise we will be adding an unnecassary 0. 
					isID = false;
				}
				else { //Add number. 
					if (number_is_negative) x = x * -1.0; //Check if negative. 
					p.coordinates.push_back(x);
					//Reset values.
					x = 0;
					comma = false;
					number_is_negative = false;
					comma_counter = 0;
				}

			}
			else if (isID) { // Is the number the ID. 
				p.ID = (p.ID *10) + (ch - 48);
			}
			else if(isdigit(ch)){ //Is it a number. 
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
			else if(ch == 46){ //Set comma flag. 
				comma = true; 
				comma_counter++; 
			}
			ch = fgetc(fp); //Get next character. 
		}

		//Final add. 
		if (number_is_negative) x = x * -1.0; 
		p.coordinates.push_back(x);
		x = 0;
		comma = false;
		comma_counter = 0;
		isID = true; 
		//Push back point.
		points.push_back(p);
	}

	for (int i = 0; i < points.size(); i++) {
		printf("Point: %d has vectors: \n", points[i].ID);
		for (int k = 0; k < points[i].coordinates.size(); k++) {
			printf("%f	", points[i].coordinates[k]);
		}
		printf("\n");
	}

	fclose(fp);
	printf("Done parsing.\n");

	return points;
}