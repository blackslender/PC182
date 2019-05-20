#include <cstdio>
#include <iostream>
#include "DatasetGPU.h"
#include <bitset>
#include <fstream>
#include <sstream>
#include "support.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

extern unsigned long long gputick;

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
bool testCudaForError()
{
	int size = 3;
	int a[3] = { 1,2,3 };
	int b[3] = { 1,2,3 };
	int c[3] = { 0,0,0 };
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Check output value
	if (!(c[0] == 2 && c[1] == 4 && c[2] == 6)) cudaStatus = cudaErrorInvalidValue;

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus == cudaSuccess;
}


DatasetGPU::DatasetGPU(int maxRecords) {
	bool success = true;
	recordCount = new int(0);
	attrCount = new int(0);
	attributesIndex = new map<string, int>();
	attributesList = new vector<string>();
	data = new int*[maxRecords];
	for (int i = 0; i < maxRecords; i++) 
		if (cudaMalloc(&(data[i]), SETSIZE * sizeof(int)) != cudaSuccess) success = false;
	
	if (cudaMalloc(&_data, maxRecords * sizeof(int*)) != cudaSuccess) success = false;
	if (cudaMemcpy(_data, data, maxRecords * sizeof(int*), cudaMemcpyHostToDevice) != cudaSuccess) success = false;
	if (!success) cout << "Error while working with GPU (Constructor)n";
}

bool DatasetGPU::newRecord(const int* recordRow) {
	int rowIndex = (*recordCount)++;
	cudaError_t e = cudaMemcpy(data[rowIndex], recordRow, SETSIZE * sizeof(int), cudaMemcpyHostToDevice);
	return (e == cudaSuccess);
}

bool DatasetGPU::newRecord(set<string> &recordSet) {
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

int* DatasetGPU::recordSetToBit(set<string> &recordSet) {
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

set<string>* DatasetGPU::bitToRecordSet(int arr[]) {
	set<string>* s = new set<string>();
	for (int i = 0; i < attributesList->size(); i++)
		if (getbit(arr, i)) s->insert(attributesList->at(i));
	return s;
}

int DatasetGPU::newAttribute(string attrName) {
	(*attributesIndex)[attrName] = (*attrCount)++;
	attributesList->push_back(attrName);
	return (*attrCount - 1);
}


int* DatasetGPU::getRecord(int recordIndex) {
	int *currentRecord = new int[SETSIZE];
	cudaError_t e = cudaMemcpy(currentRecord, data[recordIndex], SETSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (e != cudaSuccess) cout << "Error while working with GPU (get record)...\n";
	return currentRecord;
}

// Calculate support parallely
//		@_re: pointer to the record to be check
//		@_check: marking array
//		@_data: DatasetGPU
__global__ void calSupport(int* _re, char* _check, int** _data) {
	int idx = blockIdx.x;
	int i = threadIdx.x;
	_check[idx] = 1;
	int q = _data[idx][i] & _re[i];
	if (q != _re[i]) _check[idx] = 0;

}

double DatasetGPU::supportRate(set<string> &record) {
	int* re = recordSetToBit(record);
	char* check = new char[*recordCount];
	int *_re; cudaMalloc(&_re, sizeof(int)*SETSIZE);
	char *_check; cudaMalloc(&_check, (*recordCount) * sizeof(char));
	cudaError_t e;
	e = cudaMemcpy(_re, re, SETSIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (e != cudaSuccess)  cout << "Error while working with GPU (support rate)...\n";
	calSupport << <*recordCount, SETSIZE >> > (_re, _check, _data);
	e = cudaMemcpy(check, _check, (*recordCount) * sizeof(char), cudaMemcpyDeviceToHost);
	if (e != cudaSuccess) cout << "Error while working with GPU (support rate)...\n";
	int suppCount = 0;
	for (int i = 0; i < *recordCount; i++)
		if (check[i] == 1) suppCount++;
	return 1.0*suppCount / (*recordCount);
}

double DatasetGPU::supportRate(int* record) {
	int* re = record;
	char* check = new char[*recordCount];
	int *_re; cudaMalloc(&_re, sizeof(int)*SETSIZE);
	char *_check; cudaMalloc(&_check, (*recordCount) * sizeof(char));
	cudaError_t e;
	e = cudaMemcpy(_re, re, SETSIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (e != cudaSuccess)  cout << "Error while working with GPU (support rate)...\n";
	calSupport << <*recordCount, SETSIZE >> > (_re, _check, _data);
	e = cudaMemcpy(check, _check, (*recordCount) * sizeof(char), cudaMemcpyDeviceToHost);
	if (e != cudaSuccess) cout << "Error while working with GPU (support rate)...\n";
	int suppCount = 0;
	for (int i = 0; i < *recordCount; i++)
		if (check[i] == 1) suppCount++;
	return 1.0*suppCount / (*recordCount);
}

double DatasetGPU::confidenceRate(set<string> &lhsSet, set<string> &rhsSet) {
	double s1 = supportRate(lhsSet);
	set<string> s;
	s.insert(lhsSet.begin(), lhsSet.end());
	s.insert(rhsSet.begin(), rhsSet.end());
	double s2 = supportRate(s);
	return s2 / s1;
}

double DatasetGPU::confidenceRate(int*lhsSet, int*rhsSet) {
	double s1 = supportRate(lhsSet);
	int* s = new int[SETSIZE];
	for (int i = 0; i < SETSIZE; i++) s[i] = lhsSet[i] | rhsSet[i];
	double s2 = supportRate(s);
	delete[] s;
	return s2 / s1;
}

DatasetGPU* DatasetGPU::readCSV(string filename) {
	ifstream iF;
	iF.open(filename, ios::in);

	string line;
	getline(iF, line);
	DatasetGPU *d = new DatasetGPU(DEFAULT_RECORDS_COUNT);
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