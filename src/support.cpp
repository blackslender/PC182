#include "support.h"

void setbit(int arr[], int index, bool value) {
	int arrIndex = 0;
	while (index > sizeof(int)) {
		index -= sizeof(int);
		arrIndex++;
	}
	int q = 1 << index;

	if (value) arr[arrIndex] |= q;
	else arr[arrIndex] &= (!q);
}

bool getbit(int arr[], int index) {
	int arrIndex = 0;
	while (index > sizeof(int)) {
		index -= sizeof(int);
		arrIndex++;
	}
	int q = 1 << index;

	q &= arr[arrIndex];
	return q != 0;
}

int bitcount(int arr[]) {
	int ind = 0;
	int pos = 1;
	int res = 0;
	while (ind < SETSIZE) {
		if ((arr[ind] & pos) != 0)  res++;
		pos <<= 1;
		if (pos == 0) {
			pos = 1;
			ind++;
		}
	}
	return res;
}