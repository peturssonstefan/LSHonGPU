#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gloveparser.cuh"

struct PointDistance {
	int ID; 
	float distance; 
};

struct QueryPointDistances {
	int ID; 
	PointDistance *distances; 
};

thrust::device_vector<QueryPointDistances> scan(thrust::device_vector<Point>& data, thrust::device_vector<Point>& queries, int k, int dimensions);
