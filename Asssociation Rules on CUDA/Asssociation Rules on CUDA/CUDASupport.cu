
#include "CUDASupport.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void fillZero(void* arr, int size) {
	char* _arr = (char*)arr;
	int tIndex = blockIdx.x;
	if (tIndex < size) _arr[tIndex] = 0;
}

__global__ void copy(int* src, int* des, int size) {
	int tIndex = blockIdx.x;
	if (tIndex < size) des[tIndex] = src[tIndex];
}

__global__ void getBit(int *buffer, int index, bool *ret) {
	int i = index / 32;
	int offset = index % 32;
	int temp = 1 << offset;
	temp &= buffer[i];
	*ret = temp != 0;
}

bool getBit(int*buffer, int index) {
	bool ret = false;
	getBit << <1, 1 >> > (buffer, index, &ret);
	return ret;
}

__global__ void setBit(int *buffer, int index, bool *ret) {
	int i = index / 32;
	int offset = index % 32;
	int temp = 1 << offset;
	int temp2 = temp& buffer[i];
	buffer[i] &= temp;
	*ret = temp2 != 0;
}

bool setBit(int*buffer, int index) {
	bool ret = false;
	setBit << <1, 1 >> > (buffer, index, &ret);
	return ret;
}

using namespace cudasupport;

bool IntSet::insert(int value) {
	// TODO: complete this function
	return true;
}

void IntSet::print() {
	// TODO: complete this function
	return;
}

void dummyTest() {
	int *a = new int[100];
	for (int i = 0; i < 100; i++) a[i] = 1;
	int *_a; cudaMalloc(&_a, 100 * sizeof(int));
	cudaMemcpy(_a, a, 100 * sizeof(int), cudaMemcpyHostToDevice);
	fillZero << <100, 1 >> > (_a, sizeof(int) * 100);
	cudaMemcpy(a, _a, 100 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++) std::cout << a[i] << std::endl;
}