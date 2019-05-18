#ifndef CUDA_SUPPORT_H
#define CUDA_SUPPORT_H

namespace cudasupport {
	// This class is used to contain integer
	class IntSet {
		int* marker;
		int capacity;
	public:
		IntSet() {
			cudaMalloc(&marker,1);
			capacity = 0;
		}
		bool insert(int value);

		void print();
	};

	void dummyTest();
}


#endif