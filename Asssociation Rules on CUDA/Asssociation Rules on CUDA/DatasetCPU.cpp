#include <cstdio>
#include <iostream>
#include "DatasetCPU.h"
#include <bitset>
#include <fstream>
#include <sstream>
#include "support.h"

using namespace std;

DatasetCPU::DatasetCPU(int maxRecords) {
	bool success = true;
	recordCount = new int(0);
	attrCount = new int(0);
	attributesIndex = new map<string, int>();
	attributesList = new vector<string>();
	data = new int*[maxRecords];
	for (int i = 0; i < maxRecords; i++)
		/*if (cudaMalloc(&(data[i]), SETSIZE * sizeof(int)) != cudaSuccess) success = false;*/
		data[i] = new int[SETSIZE];
}

bool DatasetCPU::newRecord(const int* recordRow) {
	int rowIndex = (*recordCount)++;
	// memcpy(data[rowIndex], recordRow, SETSIZE * sizeof(int));
	for (int i=0;i<SETSIZE;i++) data[rowIndex][i] = recordRow[i];
	return true;
}

bool DatasetCPU::newRecord(set<string> &recordSet) {
	int *currentRecord = new int[SETSIZE];
	for (int i = 0; i < SETSIZE; i++) currentRecord[i] = 0;
	set<string>::iterator it;
	for (it = recordSet.begin(); it != recordSet.end(); it++) {
		map<string, int>::iterator jt = attributesIndex->find(*it);
		int idx;
		if (jt == attributesIndex->end())
			idx = newAttribute(*it);
		else idx = jt->second;
		setbit(currentRecord, idx, 1);
	}

	bool result = newRecord(currentRecord);
	delete[] currentRecord;
	return result;
}

int* DatasetCPU::recordSetToBit(set<string> &recordSet) {
	int *currentRecord = new int[SETSIZE];
	for (int i = 0; i < SETSIZE; i++) currentRecord[i] = 0;
	set<string>::iterator it;
	for (it = recordSet.begin(); it != recordSet.end(); it++) {
		map<string, int>::iterator jt = attributesIndex->find(*it);
		int idx;
		if (jt == attributesIndex->end())
			idx = newAttribute(*it);
		else idx = jt->second;
		setbit(currentRecord, idx, 1);
	}
	return currentRecord;
}

set<string>* DatasetCPU::bitToRecordSet(int arr[]) {
	set<string>* s = new set<string>();
	for (int i = 0; i < attributesList->size(); i++)
		if (getbit(arr, i)) s->insert(attributesList->at(i));
	return s;
}

int DatasetCPU::newAttribute(string attrName) {
	(*attributesIndex)[attrName] = (*attrCount)++;
	attributesList->push_back(attrName);
	return (*attrCount - 1);
}

int* DatasetCPU::getRecord(int recordIndex) {
	int *currentRecord = new int[SETSIZE];
	// memcpy(currentRecord, data[recordIndex], SETSIZE * sizeof(int));
	for (int i=0;i<SETSIZE;i++) currentRecord[i] = data[recordIndex][i];
	return currentRecord;
}

// Calculate support parallely
//		@_re: pointer to the record to be check
//		@_check: marking array
//		@_data: DatasetCPU
void calSupport(int blockIdx,int threadIdx, int* _re, char* _check, int** _data) {
	int idx = blockIdx;
	int i = threadIdx;
	int q = _data[idx][i] & _re[i];
	if (q != _re[i]) _check[idx] = 0;
}

double DatasetCPU::supportRate(set<string> &record) {
	int* re = recordSetToBit(record);
	char* check = new char[*recordCount];
	for (int i = 0; i < *recordCount; i++) check[i] = 1;
	for (int blockIdx = 0; blockIdx < *recordCount; blockIdx++)
		for (int threadIdx = 0; threadIdx < SETSIZE; threadIdx++)
			calSupport(blockIdx, threadIdx, re, check, data);
	int suppCount = 0;
	for (int i = 0; i < *recordCount; i++)
		if (check[i] == 1) suppCount++;
	return 1.0*suppCount / (*recordCount);
}

double DatasetCPU::supportRate(int* record) {
	int* re = record;
	char* check = new char[*recordCount];
	for (int i = 0; i < *recordCount; i++) check[i] = 1;
	for (int blockIdx = 0; blockIdx < *recordCount; blockIdx++)
		for (int threadIdx = 0; threadIdx < SETSIZE; threadIdx++)
			calSupport(blockIdx, threadIdx, re, check, data);
	int suppCount = 0;
	for (int i = 0; i < *recordCount; i++)
		if (check[i] == 1) suppCount++;
	return 1.0*suppCount / (*recordCount);
}

double DatasetCPU::confidenceRate(set<string> &lhsSet, set<string> &rhsSet) {
	double s1 = supportRate(lhsSet);
	set<string> s;
	s.insert(lhsSet.begin(), lhsSet.end());
	s.insert(rhsSet.begin(), rhsSet.end());
	double s2 = supportRate(s);
	return s2 / s1;
}

double DatasetCPU::confidenceRate(int*lhsSet, int*rhsSet) {
	double s1 = supportRate(lhsSet);
	int* s = new int[SETSIZE];
	for (int i = 0; i < SETSIZE; i++) s[i] = lhsSet[i] | rhsSet[i];
	double s2 = supportRate(s);
	delete[] s;
	return s2 / s1;
}

DatasetCPU* DatasetCPU::readCSV(string filename) {
	ifstream iF;
	iF.open(filename, ios::in);

	string line;
	getline(iF, line);
	DatasetCPU *d = new DatasetCPU(DEFAULT_RECORDS_COUNT);
	set<string> attributesSet;

	// Read attribute names
	stringstream ss(line);
	string sname;
	getline(ss, sname, ','); // Skip the first "name" thing
	while (getline(ss, sname, ',')) {
		d->newAttribute(sname);
	}

	// Now read records
	while (getline(iF, line)) {
		stringstream ss(line);
		string value;
		set<string> currentRecord;
		int index = 0;
		getline(ss, sname, ','); // Skip the record name
		while (getline(ss, value, ',')) {
			if (value == "y" || value == "Y" || value == "1")
				currentRecord.insert(d->getAttributesSet()->at(index));
			index++;
		}

		// Print current record

		if (!d->newRecord(currentRecord)) cout << "Error while working with GPU (inserting record)...\n";
	}
	return d;
}