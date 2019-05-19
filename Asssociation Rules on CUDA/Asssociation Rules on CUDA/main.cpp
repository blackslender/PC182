#include <iostream>
#include <vector>
#include <string>
#include <set>
#include "support.h"
#include <bitset>
#include <queue>
#include <utility>
using namespace std;

#define SUP_RATE 0.3999
#define CONFIDENCE 0.9

int main(int argc, char **argv) {

	if (!testCudaForError()) {
		cout << "Error while working with GPU...\n";
		return -1;
	}

	Dataset *d = readCSV("data.csv");

	vector<string>& attributes = *(d->getAttributesSet());
	set<set<string>> frequentSets;
	queue<pair<set<string>, int>> L;

	for (int i = 0; i < attributes.size(); i++) {
		pair<set<string>, int> p;
		set<string> s;
		s.insert(attributes[i]);
		p.first = s;
		p.second = i + 1;
		L.push(p);
	}

	while (!L.empty()) {

		set<string> &currentSet = L.front().first;
		int &ind = L.front().second;

		double support = d->supportRate(currentSet);
		if (support > SUP_RATE) {
			frequentSets.insert(currentSet);
			for (int i = ind; i < attributes.size(); i++) {
				set<string> newSet = currentSet;
				newSet.insert(attributes[i]);
				pair<set<string>, int> p;
				p.first = newSet;
				p.second = i + 1;
				L.push(p);
			}
		}
		L.pop();
	}
	set<set<string>>::iterator it;
	for (it = frequentSets.begin(); it != frequentSets.end(); it++) {
		set<string>::iterator jt;
		for (jt = it->begin(); jt != it->end(); jt++)
			cout << *jt << " ";
		cout << endl;
	}

	return 0;
}


