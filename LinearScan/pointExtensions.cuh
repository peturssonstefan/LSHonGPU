#include "point.h"
#include "math.h"
#include <device_launch_parameters.h>

__inline__ __device__
Point min(Point p1, Point p2) {
	return p1.distance < p2.distance ? p1 : p2; 
}

__inline__ __device__
Point max(Point p1, Point p2) {
	return p1.distance > p2.distance ? p1 : p2;
}

__inline__ __device__
Point createPoint(int ID, float distance) {
	Point p; 
	p.ID = ID; 
	p.distance = distance; 
	return p;
}