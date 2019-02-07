#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "linearScan.cuh"

//Angular distance functor.
struct agFunctor {
	thrust::device_vector<Point> data;
	const int k; 

	agFunctor(thrust::device_vector<Point>& _data, int _k) : data(_data), k(_k) {}

	__host__ __device__
		QueryPointDistances operator()(const Point& q) {
			QueryPointDistances qpd; 
			qpd.ID = q.ID; 
			qpd.distances = (PointDistance*)malloc(k * sizeof(PointDistance)); 

			for (int i = 0; i < data.size(); i++) {
				if (i < k) {
					Point p = data[i];
					PointDistance pointDistance;
					pointDistance.ID = p.ID;
					int index = i % 5;
					pointDistance.distance = 5;
				}
			}
			return qpd;
		}
};

thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k) {
	printf("Starting scan... \n");
	thrust::device_vector<QueryPointDistances> items; 
	thrust::transform(queries.begin(), queries.end(), items.begin(), agFunctor(data, k));

	printf("Items size: %d", items.size());
	
	return items; 
}