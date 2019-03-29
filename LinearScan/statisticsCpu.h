#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <map>
#include <bitset>
#include <iostream>
#include <string>

std::map<std::string, int> bucketDistributionFullKey(unsigned char* hashes, int hashesSize, int sketchDim);