#include <stdio.h>
#include <iostream>
#include <string>
#include "resultWriter.h"

void writeResult(Point* results, int k, int n_query, int reportK) {
	
	std::cout << n_query << std::endl;
	std::cout << reportK << std::endl;
	
	for (int i = 0; i < n_query; i++) {
		printf("%d:\n", i); 
		for (int j = 0; j < reportK; j++) {
			Point p = results[i*k + j]; 
			printf("%d %f\n", p.ID, p.distance);
		}
	}

}


void writeOnlyIDs(Point* results, int k, int n_query) {

	std::cout << n_query << std::endl;
	std::cout << k << std::endl;

	for (int i = 0; i < n_query; i++) {
		for (int j = 0; j < k; j++) {
			Point p = results[i*k + j];
			printf("%d\n", p.ID);
		}
	}

}