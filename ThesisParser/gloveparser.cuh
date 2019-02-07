#include <thrust/host_vector.h>

struct Point {
	int ID;
	thrust::host_vector<float> coordinates;
};

thrust::host_vector<Point> parseFile(char* path, int dimensions);