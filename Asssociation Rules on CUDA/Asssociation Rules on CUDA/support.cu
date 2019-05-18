#include <cstdio>
#include <iostream>
#include "support.h"
#include <bitset>

using namespace std;

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

void setbit(int arr[], int index, bool value) {
	int arrIndex = 0;
	while (index > sizeof(int)) {
		index -= sizeof(int);
		arrIndex++;
	}
	int q = 1 << index;

	arr[arrIndex] ^= q;
}

Dataset::Dataset(int maxRecords) {
	recordCount = new int(0);
	attrCount = new int(0);
	attributesIndex = new map<string, int>();
	syncHostToDevice();
	data = new int*[maxRecords];
	for (int i = 0; i < maxRecords; i++)
		cudaMalloc(&(data[i]), SETSIZE * sizeof(int));
	cudaMalloc(&_data, maxRecords * sizeof(int*));
	cudaMemcpy(_data, data, maxRecords * sizeof(int*), cudaMemcpyHostToDevice);
}

bool Dataset::newRecord(const int* recordRow) {
	int rowIndex = (*recordCount)++;
	cudaError_t e = cudaMemcpy(data[rowIndex], recordRow, SETSIZE * sizeof(int), cudaMemcpyHostToDevice);
	syncHostToDevice();
	return (e == cudaSuccess);
}

bool Dataset::newRecord(set<string> &recordSet) {
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
	for (int i = 0; i < SETSIZE; i++) {
		bitset<32> x(currentRecord[i]);
		cout << x;
	} cout << endl;
	bool result = newRecord(currentRecord);
	delete[] currentRecord;
	return result;
}

int* Dataset::recordSetToBit(set<string> &recordSet) {
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

int Dataset::newAttribute(string attrName) {
	(*attributesIndex)[attrName] = (*attrCount)++;
	syncHostToDevice();
	return (*attrCount - 1);
}

bool Dataset::syncHostToDevice() {
	cudaError_t e = cudaMemcpy(recordCount, _recordCount, sizeof(int), cudaMemcpyHostToDevice);
	if (e != cudaSuccess) return false;
	e = cudaMemcpy(attrCount, _attrCount, sizeof(int), cudaMemcpyHostToDevice);
	if (e != cudaSuccess) return false;
	return true;
}

int* Dataset::getRecord(int recordIndex) {
	int *currentRecord = new int[SETSIZE];
	cudaError_t e = cudaMemcpy(currentRecord, data[recordIndex], SETSIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

// Calculate support parallely
//		@_re: pointer to the record to be check
//		@_check: marking array
//		@_data: dataset
__global__ void calSupport(int* _re, char* _check, int** _data) {
	int idx = blockIdx.x;
	
}

double Dataset::supportRate(set<string> &record) {
	int* re = recordSetToBit(record);
	char* check = new char[*recordCount];

	int *_re; cudaMalloc(&re, sizeof(int)*SETSIZE);
	int *_check; cudaMalloc(&re, (*recordCount) * sizeof(char));
}