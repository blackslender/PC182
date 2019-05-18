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

	// Insert new record
	//		@recordRow: record represented as a binary array
	//		Return: true if record is successfully inserted, otherwise false
	bool newRecord(const int* recordRow);

	// New attribute
	//		@attrName: a string represents for the attribute
	//		Return: index of the new attribute to be stored
	int newAttribute(string attrName);

	// Sync values from host to device
	// This should be called before using any GPU method
	bool syncHostToDevice();

	// Get a binary array represents for the index
	//		@recordIndex: index of the record in dataset
	// THIS FUNCTION SHOULDN'T BE USED, JUST FOR TESTING
	int* getRecord(int recordIndex);

	// This function is used to convert a record represented as set to a binary array
	//		@recordSet: set of attribute references to the record
	//		Return: Pointer to the binary array
	// HEAP VALUE SHOULD BE CLEAN AFTER USING
	int* Dataset::recordSetToBit(set<string> &recordSet);

public:

	// @maxRecords: Number of record to be initialized on GPU ram
	Dataset(int maxRecords);

	// Insert new record
	//		@recordSet: set<string> of records, each string represents an attribute
	//		Return: true if record is successfully inserted, otherwise false
	bool newRecord(set<string> &recordSet);

	// Calculate the support rate of a record over the dataset
	//		@recordSet: set<string> of records, each string represents an attribute
	double supportRate(set<string> &recordSet);
};

#endif