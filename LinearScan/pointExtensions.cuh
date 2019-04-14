#pragma once
#include "point.h"
#include "math.h"
#include <device_launch_parameters.h>
#include "pointWithIndex.h"
#include "constants.cuh"

__inline__ __device__
bool same(Point p1, Point p2) {
	return (abs(p1.distance - p2.distance) < EPSILON);
}

__inline__ __device__
Point min(Point p1, Point p2) {
	if (same(p1, p2)) {
		return p1;
	}
	return p1.distance < p2.distance ? p1 : p2; 
}

__inline__ __device__
Point max(Point p1, Point p2) {
	if (same(p1, p2)) {
		return p1;
	}
	return p1.distance > p2.distance ? p1 : p2;
}

__inline__ __device__
Point createPoint(int ID, float distance) {
	Point p; 
	p.ID = ID; 
	p.distance = distance; 
	return p;
}


__inline__ __device__
PointWithIndex min(PointWithIndex p1, PointWithIndex p2) {
	return p1.idx < p2.idx ? p1 : p2;
}

__inline__ __device__
PointWithIndex max(PointWithIndex p1, PointWithIndex p2) {
	return p1.idx > p2.idx ? p1 : p2;
}

__inline__ __device__
PointWithIndex createPointWithIndex(int idx, Point p) {
	PointWithIndex pWI;

	pWI.idx = idx;
	pWI.ID = p.ID;
	pWI.distance = p.distance;

	return pWI;
}
