#include <math.h>
#include <iostream>
#include <random>

float* generateRandomVectors(int N, bool randomSeed) {

	// same seed 
	printf("Creating random vectors \n");
	static float* vectors = (float*)malloc(N * sizeof(float));
	std::default_random_engine generator;
	// different seeds
	printf("Setup array and generator \n");
	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator

	std::normal_distribution<double> distribution(0.0, 1.0); // Standard normal distribution.

	printf("Starting loop \n");
	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		//std::cout << vectors[i] << ",";
	}

	printf("Returning vectors \n");
	//std::cout << std::endl; 
	return vectors;
}


void generateRandomVectors(float* vectors, int N, bool randomSeed) {

	// same seed 
	printf("Creating random vectors \n");
	std::default_random_engine generator;
	// different seeds
	printf("Setup array and generator \n");
	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator

	std::normal_distribution<double> distribution(0.0, 1.0); // Standard normal distribution.

	printf("Starting loop \n");
	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		//std::cout << vectors[i] << ",";
	}

	printf("Returning vectors \n");
	//std::cout << std::endl; 
}


float* generateRandomVectors(int N, int sketchDim, bool randomSeed) {

	// same seed 
	printf("sketchDim in random vec: %d \n", sketchDim);
	static float* vectors = (float*)malloc(N * sizeof(float));
	std::default_random_engine generator;
	// different seeds
	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator

	std::normal_distribution<double> distribution(0.0, 1.0 / sketchDim); // Standard normal distribution.

	for (int i = 0; i < N; ++i)
	{
		vectors[i] = distribution(randomSeed ? eng : generator);
		//std::cout << vectors[i] << ",";
	}
	//std::cout << std::endl; 
	return vectors;
}