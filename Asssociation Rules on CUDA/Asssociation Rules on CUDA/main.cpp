#include <iostream>
#include <vector>
#include <string>
#include <set>
#include "support.h"

using namespace std;

#define SUP_RATE 0.6
#define CONFIDENCE 0.9

int main(int argc, char **argv) {

	if (!testCudaForError()) cout << "Error while working with GPU...\n";

	Dataset *d = new Dataset(16);

	set<string> s;
	s.insert("attr1");
	s.insert("attr2");
	s.insert("attr3");
	d->newRecord(s);
	set<string> s2,s3;
	s2.insert("attr2");
	s2.insert("attr3");
	d->newRecord(s2);
	s3.insert("atr4");
	d->newRecord(s3);
	return 0;
}


