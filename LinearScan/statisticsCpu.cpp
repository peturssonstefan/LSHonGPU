#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <map>
#include <bitset>
#include <iostream>
#include "statisticsCpu.h"

using namespace std; 

map<string, int> bucketDistributionFullKey(unsigned char* hashes, int hashesSize, int sketchDim) {
	
	map<string, int> m; 
	for (int i = 0; i < hashesSize; i++) {
		string hash = ""; 
		for (int dim = 0; dim < sketchDim; dim++) {
			hash += bitset<8>(hashes[i * sketchDim + dim]).to_string();
		}
		cout << hash << endl;
		
		if (m.find(hash) != m.end()) {
			m[hash]++; 
		}
		else {
			m[hash] = 1; 
		}
	}
}

