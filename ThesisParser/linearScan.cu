#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "linearScan.cuh"


thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k, int dimensions) {
	printf("Starting scan... \n");
	
	for (Point query : queries) {
		thrust::device_vector<float> queryCoords(query.coordinates, query.coordinates + dimensions);

		for (Point point : data) {
			thrust::device_vector<float> pointCoords(point.coordinates, point.coordinates + dimensions);

			thrust::transform(pointCoords.begin(), pointCoords.end(), queryCoords.begin(), pointCoords.begin(), thrust::multiplies<float>());
			float innerProduct = thrust::reduce(pointCoords.begin(), pointCoords.end(), (float)0.0, thrust::plus<float>());

			printf("The inner product between %d - %d is: %f \n", query.ID, point.ID, innerProduct);
		}
	}

	
	thrust::device_vector<QueryPointDistances> items; 
	
	
	cudaDeviceSynchronize();

	return items;
	
}