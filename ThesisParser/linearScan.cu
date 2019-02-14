#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "linearScan.cuh"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
//
//struct testFunctor {
//
//	__host__ __device__
//		thrust::device_vector<float> operator()(thrust::device_vector<float> data, thrust::device_vector<float> query) {
//			thrust::transform(data.begin(), data.end(), query.begin(),data.begin(), thrust::plus<float>());
//			return data;
//	}
//
//};
//
//
//void multiply(float *data, const float *query, int d, int n) {
//	for (int i = 0; i < n; i++) {
//		for (int j = 0; j < d; j++) {
//			data[j + i * d] = data[j + i * d] + query[j];
//		}
//	}
//}
//
//thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k, int dimensions) {
//	thrust::device_vector<thrust::device_vector<float>> dataTmp;
//
//	thrust::device_vector<QueryPointDistances> queryPointDistances; 
//
//	for (int i = 0; i < data.size(); i++) {
//		thrust::device_vector<float> list; 
//		for (int j = 0; j < dimensions; j++) {
//			Point p = data[i]; 
//			list.push_back(p.coordinates[j]);
//		}
//		dataTmp.push_back(list);
//	}
//
//
//	Point queryPoint = queries[0]; //test query point.
//	thrust::device_vector<float> query;
//	for (int i = 0; i < dimensions; i++) {
//		query.push_back(queryPoint.coordinates[i]);
//	}
//
//	//thrust::constant_iterator<thrust::device_vector<float>> constant_query(query);
//
//	//thrust::transform(dataTmp.begin(), dataTmp.end(), constant_query, data.begin(), testFunctor());
//
//	for (int i = 0; i < 10; i++) {
//		printf("Vector %d: \n", i);
//		for (int j = 0; j < dimensions; j++) {
//			thrust::device_vector<float> tmp = dataTmp[i];
//			printf("%f - ", tmp[j]); 
//		}
//		printf("\n \n");
//	}
//
//	return queryPointDistances;
//
//}


thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k, int dimensions) {
	thrust::device_vector<QueryPointDistances> queryPointDistances;
	return queryPointDistances;
}