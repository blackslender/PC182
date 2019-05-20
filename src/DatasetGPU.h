#ifndef DATASET_GPU_H
#define DATASET_GPU_H

#include <vector>
#include <map>
#include <string>
#include <set>

using namespace std;

bool testCudaForError();

class DatasetGPU {
	int** data;
	int** _data;
	int *recordCount;
	int *attrCount;
	map<string, int> *attributesIndex;
	vector<string> *attributesList;

	DatasetGPU() {

	}

	// Insert new record
	//		@recordRow: record represented as a binary array
	//		Return: true if record is successfully inserted, otherwise false
	bool newRecord(const int* recordRow);

public:

	// @maxRecords: Number of record to be initialized on GPU ram
	DatasetGPU(int maxRecords);

	// Insert new record
	//		@recordSet: set<string> of records, each string represents an attribute
	//		Return: true if record is successfully inserted, otherwise false
	bool newRecord(set<string> &recordSet);

	// Calculate the support rate of a record over the DatasetGPU
	//		@recordSet: set<string> of record, each string represents an attribute
	double supportRate(set<string> &recordSet);

	// Calculate the support rate of a record over the DatasetGPU
	//		@recordBit: binary bit of record, each string represents an attribute
	double supportRate(int* recordSet);

	// Calculate the confidendce rate of a rule over the DatasetGPU
	//		@lhsSet: set<string> of lhs attributes, each string represents an attribute
	//		@rhsSet: set<string> of rhs attributes, each string represents an attribute
	double confidenceRate(set<string> &lhsSet, set<string> &rhsSet);

	// Calculate the confidendce rate of a rule over the DatasetGPU
	//		@lhsSet: binary bit array of lhs attributes, each string represents an attribute
	//		@rhsSet: binary bit array rhs attributes, each string represents an attribute
	double confidenceRate(int*lhsSet, int*rhsSet);

	// Get a binary array represents for the index
	//		@recordIndex: index of the record in DatasetGPU
	// THIS FUNCTION SHOULDN'T BE USED, JUST FOR TESTING
	int* getRecord(int recordIndex);

	// New attribute
	//		@attrName: a string represents for the attribute
	//		Return: index of the new attribute to be stored
	int newAttribute(string attrName);

	vector<string> *getAttributesSet() { return attributesList; }

	// This function is used to convert a record represented as set to a binary array
	//		@recordSet: set of attribute references to the record
	//		Return: Pointer to the binary array
	// HEAP VALUE SHOULD BE CLEAN AFTER USING
	int* recordSetToBit(set<string> &recordSet);

	set<string>* bitToRecordSet(int arr[]);

	int getRecordCount() {
		return *recordCount;
	}

	static DatasetGPU* readCSV(string filename);

};



#endif