#pragma once 
#include "point.h"
#include "pointExtensions.cuh"
#include <cuda.h>
#include <bitset>
#include <math.h>
#include <time.h>
#include "constants.cuh"
#include "sketchedDistanceScanners.cuh"
#include "launchHelper.cuh"
#include "cudaHelpers.cuh"
#include "randomVectorGenerator.h"
#include "resultDTO.h"

namespace crosspoly {
	
	template<class T>__global__
	void normalizeVectors(LaunchDTO<T> launchDTO) {
		transformToUnitVectors(launchDTO.queries, launchDTO.N_queries, launchDTO.dimensions);
		transformToUnitVectors(launchDTO.data, launchDTO.N_data, launchDTO.dimensions);
	}

	

	template<class T, class K>__global__ 
	void generateHashCrossPoly(LaunchDTO<T> params, float* originalData, K* sketchedData,int N_data, int sketchDim, int* randomDiagonals, float* partialResults, float* tempResults, int partialSize) {
		int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
		int totalNumberOfThreads = gridDim.x * blockDim.x;

		int partialResultIdx = threadId * partialSize;
		int randDiaFullSize = 3 * partialSize; 

		for (int i = threadId; i < N_data; i += totalNumberOfThreads) {
			int dataIdx = i * params.dimensions;
			
			for (int sketchDimIdx = 0; sketchDimIdx < sketchDim; sketchDimIdx++) {


				for (int pI = 0; pI < partialSize; pI++) {
					partialResults[partialResultIdx + pI] = pI < params.dimensions ? originalData[dataIdx + pI] : 0;
				}

				
				for (int randId = 0; randId < 3; randId++) {
					for (int dimId = 0; dimId < partialSize; dimId++) {
						
						partialResults[partialResultIdx + dimId] *= (float)randomDiagonals[sketchDimIdx * 3 * partialSize + (randId*partialSize + dimId)];
					}

					for (int offset = partialSize / 2; offset > 0; offset /= 2) {
						for (int dimId = 0; dimId < partialSize; dimId++) {
							int otherId = dimId ^ offset;

							tempResults[partialResultIdx + dimId] = dimId < otherId ?
								partialResults[partialResultIdx + dimId] + partialResults[partialResultIdx + otherId] :
								-partialResults[partialResultIdx + dimId] + partialResults[partialResultIdx + otherId];

						}

						for (int dimId = 0; dimId < partialSize; dimId++) {
							partialResults[partialResultIdx + dimId] = tempResults[partialResultIdx + dimId];
						}
					}
				}

				int index;
				float totalMax = float(INT_MIN);
				// find base vector
				for (int dimId = 0; dimId < partialSize; dimId++) {
					bool negative = partialResults[partialResultIdx + dimId] < 0;
					//printf("PSize = %d T[%d] PartialRes[%d]: %f\n",partialSize, threadId, dimId,partialResults[partialResultIdx + dimId]); 
					float maxVal = negative ? max(totalMax, -1*partialResults[partialResultIdx + dimId]) : max(totalMax , partialResults[partialResultIdx + dimId]);
					if (maxVal > totalMax) {
						index = negative ? dimId : dimId + partialSize;
						totalMax = maxVal;
					}
				}

				//printf("Query %d has index %d \n", threadId, index); 
				sketchedData[i * sketchDim + sketchDimIdx] = index; 
			}

		}

	}

	bool isPowerOfTwo(int n)
	{
		if (n == 0)
			return false;

		return (ceil(log2(n)) == floor(log2(n)));
	}

	int logDim(int dim) {
		if (isPowerOfTwo(dim)) return dim; 
		else {
			int logTwoDim = 1; 
			while (logTwoDim < dim) {
				logTwoDim *= 2; 
			}
			return logTwoDim; 
		}
	}

	template <class T, class K>
	void crossPolyTopeHashing(LaunchDTO<T> params, LshLaunchDTO<K> lshParams, Result& res, int numberOfBlocks, int numberOfThreads) {
		time_t before = clock(); 
		normalizeVectors << <numberOfBlocks, numberOfThreads >> > (params);
		waitForKernel(); 
		int logDimSize = logDim(params.dimensions); 
		printf("Log dim size: %d \n", logDimSize); 
		int* randomDiagonals = (int*)malloc(logDimSize*lshParams.tables * 3 * sizeof(int));
		time_t after = clock() - before;
		res.preprocessTime = res.calcTime(after);
		before = clock(); 
		generateRandomOnePlusMinusVector(logDimSize*lshParams.tables*3, randomDiagonals); 
		int* dev_randomDiagonals = mallocArray(randomDiagonals, logDimSize*lshParams.tables * 3, true);
		
		int totalThreads = numberOfBlocks * numberOfThreads; 
		float* partialResults = (float*)malloc(totalThreads * logDimSize * sizeof(float)); 
		float* tempResults = (float*)malloc(totalThreads * logDimSize * sizeof(float));
		float* dev_partialResults = mallocArray(partialResults, totalThreads * logDimSize);
		float* dev_tempResults = mallocArray(partialResults, totalThreads * logDimSize);


		generateHashCrossPoly <<<numberOfBlocks, numberOfThreads >>>(params, params.data, lshParams.dataKeys, params.N_data, lshParams.tables, dev_randomDiagonals, dev_partialResults, dev_tempResults, logDimSize);
		waitForKernel(); 
		generateHashCrossPoly << <numberOfBlocks, numberOfThreads >> > (params, params.queries, lshParams.queryKeys, params.N_queries, lshParams.tables, dev_randomDiagonals, dev_partialResults, dev_tempResults, logDimSize);
		waitForKernel(); 

		K* queryKeys = (K*)malloc(params.N_queries * lshParams.tables * sizeof(K)); 
		copyArrayToHost(queryKeys, lshParams.queryKeys, params.N_queries * lshParams.tables); 

		/*for (int i = 0; i < params.N_queries; i++) {
			printf("QueryID: %d \n", i); 
			for (int j = 0; j < lshParams.tables; j++) {
				printf("HashTable: %d has key %d \n", j, queryKeys[i * lshParams.tables + j]); 
			}
			printf("\n"); 
		}*/

	}


}
