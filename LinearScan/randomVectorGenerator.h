#pragma once


float* generateRandomVectors(int N, bool randomSeed = false);

float* generateRandomVectors(int N, int sketchDim, bool randomSeed = false);

void generateRandomVectors(float* vectors, int N, bool randomSeed = false);

void generateRandomOnePlusMinusVector(int size, int* vectors); 