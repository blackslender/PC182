#ifndef SUPPORT_H
#define SUPPORT_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <map>
#include <string>
#include <set>
// Maximum number to be stored in BinarySet
#define SETSIZE 16

using namespace std;

bool testCudaForError();

void setbit(int arr[], int index, bool value);

class Dataset {
	int** data;
	int** _data;
	int *recordCount;
	int *_recordCount;
	int *attrCount;
	int *_attrCount;

	map<string, int> *attributesIndex;

	Dataset() {

	}
public:
	Dataset(int maxRecords);

	bool newRecord(const int* recordRow);

	bool newRecord(set<string> &recordSet);

	int newAttribute(string attrName);

	bool syncHostToDevice();

	int* getRecord(int recordIndex);
};

#endif