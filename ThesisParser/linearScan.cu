#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "linearScan.cuh"

//Angular distance functor.
struct agFunctor : public thrust::binary_function<float, float, float> {
	const float a; 
	agFunctor(float _a): a(_a){}
	__host__ __device__
		float operator()(const float& x, const float& y) const {
			return a * x + y;
		}
};

thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k) {
	printf("Starting scan... \n");
	float x[4] = { 1.0, 1.0, 1.0, 1.0 };
	float y[4] = { 1.0, 2.0, 3.0, 4.0 };

	// transfer to device
	thrust::device_vector<float> X(x, x + 4);
	thrust::device_vector<float> Y(y, y + 4);

	// fast method
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), agFunctor(k));
	
	thrust::device_vector<QueryPointDistances> items; 
	//thrust::transform(queries.begin(), queries.end(), items.begin(), agFunctor(data, k));

	cudaDeviceSynchronize();

	for (int i = 0; i < 4; i++) {
		float tmp = Y[i]; 
		printf("%d: %f \n", i, tmp);
	}

	return items;
	
}