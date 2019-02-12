#pragma once
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

float* parseFile(char* path, int& n, int& dimensions);