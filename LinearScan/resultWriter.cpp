#include <stdio.h>
#include <iostream>
#include <fstream>
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

void writePerformanceResults(Result result, LaunchDTO<int> launchDTO, LshLaunchDTO<int> lshLaunchDTO, char* resultFile){
	std::cout << "Writing results to file " << resultFile << std::endl; 
	std::ofstream fileStream;

	fileStream.open(resultFile, std::ios_base::app);
	
	fileStream << launchDTO.implementation << ",";
	fileStream << lshLaunchDTO.keyImplementation << ",";
	fileStream << launchDTO.sketchDim << ",";
	fileStream << launchDTO.k << ",";
	fileStream << THREAD_QUEUE_SIZE << ",";
	fileStream << lshLaunchDTO.bucketKeyBits << ",";
	fileStream << lshLaunchDTO.tables << ",";
	fileStream << WITH_TQ_OR_BUFFER << ",";
	fileStream << result.preprocessTime << ",";
	fileStream << result.constructionTime << ",";
	fileStream << result.scanTime << ",";
	fileStream << result.recall << ",";
	fileStream << result.avgDistance;

	fileStream << std::endl;

	fileStream.close();
}
