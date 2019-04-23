#pragma once
#include "point.h"
#include <time.h>

struct Result
{
	Point* results;
	clock_t preprocessTime;
	clock_t constructionTime;
	clock_t scanTime;
	float recall; 
	float avgDistance; 

	void setupResult(int N_query, int k) {
		results = (Point*)malloc(N_query * k * sizeof(Point));
		preprocessTime = 0;
		constructionTime = 0;
		scanTime = 0;
		recall = 0;
		avgDistance = 0;
	}

	void copyResultPoints(Point* points, int N_query, int k) {
		for (int i = 0; i < N_query * k; i++) {
			results[i] = points[i];
		}
	}

	time_t calcTime(time_t time) {
		return (time * 1000 / CLOCKS_PER_SEC);
	}
};
