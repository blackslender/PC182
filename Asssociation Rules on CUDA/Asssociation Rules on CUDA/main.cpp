#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <bitset>
#include <queue>
#include <stack>
#include <utility>
#include <Windows.h>
#include <iomanip>
#include "support.h"
#include "DatasetCPU.h"
#include "DatasetGPU.h"

using namespace std;

unsigned long long gputick;

DatasetCPU *dCPU = NULL;
DatasetGPU *dGPU = NULL;

void findRulesGPU(set<string> &s, set<string>::iterator it, set<string> &ls, set<string> &rs) {

	if (it == s.end()) {
		if (ls.empty() || rs.empty()) return;
		double confidence = dGPU->confidenceRate(ls, rs);
		if (confidence >= CON_RATE) {
			cout << "( ";
			for (set<string>::iterator itl = ls.begin(); itl != ls.end(); itl++)
				cout << *itl << " ";
			cout << ") => ( ";
			for (set<string>::iterator itr = rs.begin(); itr != rs.end(); itr++)
				cout << *itr << " ";
			cout << "): " << confidence << endl; cout.flush();
		}
	}
	else {
		ls.insert(*it);
		findRulesGPU(s, next(it), ls, rs);
		ls.erase(*it);

		rs.insert(*it);
		findRulesGPU(s, next(it), ls, rs);
		rs.erase(*it);
	}
}

void findRulesCPU(set<string> &s, set<string>::iterator it, set<string> &ls, set<string> &rs) {

	if (it == s.end()) {
		if (ls.empty() || rs.empty()) return;
		double confidence = dCPU->confidenceRate(ls, rs);
		if (confidence >= CON_RATE) {
			cout << "( ";
			for (set<string>::iterator itl = ls.begin(); itl != ls.end(); itl++)
				cout << *itl << " ";
			cout << ") => ( ";
			for (set<string>::iterator itr = rs.begin(); itr != rs.end(); itr++)
				cout << *itr << " ";
			cout << "): " << confidence << endl; cout.flush();
		}
	}
	else {
		ls.insert(*it);
		findRulesCPU(s, next(it), ls, rs);
		ls.erase(*it);

		rs.insert(*it);
		findRulesCPU(s, next(it), ls, rs);
		rs.erase(*it);
	}
}


int runOnCPU(int argc, char **argv) {
	gputick = 0;
	unsigned long long totaltick = GetTickCount64();


	//if (!testCudaForError()) {
	//	cout << "Error while working with GPU...\n";
	//	return -1;
	//}
	dCPU = DatasetCPU::readCSV(string(argv[1]));

	vector<string>& attributes = *(dCPU->getAttributesSet());
	set<int*> frequentSets;
	stack<pair<int*, int>> L;
	for (int i = 0; i < attributes.size(); i++) {
		pair<int*, int> p;
		set<string> s;
		s.insert(attributes[i]);
		int *t = dCPU->recordSetToBit(s);
		p.first = t;
		p.second = i + 1;
		L.push(p);
	}

	int count = 0;
	while (!L.empty()) {
		int* currentSet = L.top().first;
		int &ind = L.top().second;
		double support = dCPU->supportRate(currentSet);
		if (support > SUP_RATE) {
			for (int i = ind; i < attributes.size(); i++) {
				int* newSet = new int[SETSIZE];
				memcpy(newSet, currentSet, SETSIZE * sizeof(int));
				setbit(newSet, i, 1);
				pair<int*, int> p;
				p.first = newSet;
				p.second = i + 1;
				L.push(p);
			}
			if (bitcount(currentSet) > 1) {
				frequentSets.insert(currentSet);
			}
			else delete[] currentSet;

		}
		else delete[] currentSet;
		ind = -1;
		while (!L.empty() && L.top().second == -1) { L.pop(); }
	}

	cout << "Dataset records: " << dCPU->getRecordCount() << endl; cout.flush();
	cout << "Dataset attributes: " << dCPU->getAttributesSet()->size() << endl; cout.flush();
	if (argv[2][0] == '1') {
		cout << "\nFrequent item sets: " << endl; cout.flush();
		set<int*>::iterator it;
		for (it = frequentSets.begin(); it != frequentSets.end(); it++) {
			set<string> *q = dCPU->bitToRecordSet(*it);
			for (set<string>::iterator jt = q->begin(); jt != q->end(); jt++) {
				cout << *jt << " ";
			}
			double support = dCPU->supportRate(*q);
			cout << setprecision(4) << fixed << support << endl; cout.flush();
			delete q;

		}

		cout << "\nStrong rules: " << endl; cout.flush();
		for (it = frequentSets.begin(); it != frequentSets.end(); it++) {
			set<string> *q = dCPU->bitToRecordSet(*it);
			set<string> ls, rs;
			findRulesCPU(*q, q->begin(), ls, rs);
			delete q;
		}
	}
	cout << "\nTotal time: " << (GetTickCount64() - totaltick) / 1000.0 << endl; cout.flush();
	cout << "GPU time: " << gputick / 1000.0 << endl; cout.flush();
	return 0;
}

int runOnGPU(int argc, char **argv) {
	gputick = 0;
	unsigned long long totaltick = GetTickCount64();


	if (!testCudaForError()) {
		cout << "Error while working with GPU...\n";
		return -1;
	}
	dGPU = DatasetGPU::readCSV(string(argv[1]));

	vector<string>& attributes = *(dGPU->getAttributesSet());
	set<int*> frequentSets;
	stack<pair<int*, int>> L;
	for (int i = 0; i < attributes.size(); i++) {
		pair<int*, int> p;
		set<string> s;
		s.insert(attributes[i]);
		int *t = dGPU->recordSetToBit(s);
		p.first = t;
		p.second = i + 1;
		L.push(p);
	}

	int count = 0;
	while (!L.empty()) {
		int* currentSet = L.top().first;
		int &ind = L.top().second;

		double support = dGPU->supportRate(currentSet);

		if (support > SUP_RATE) {
			for (int i = ind; i < attributes.size(); i++) {
				int* newSet = new int[SETSIZE];
				memcpy(newSet, currentSet, SETSIZE * sizeof(int));
				setbit(newSet, i, 1);
				pair<int*, int> p;
				p.first = newSet;
				p.second = i + 1;
				L.push(p);
			}
			if (bitcount(currentSet) > 1) frequentSets.insert(currentSet); else delete[] currentSet;

		}
		else delete[] currentSet;
		ind = -1;
		while (!L.empty() && L.top().second == -1) { L.pop(); }
	}

	cout << "Dataset records: " << dGPU->getRecordCount() << endl; cout.flush();
	cout << "Dataset attributes: " << dGPU->getAttributesSet()->size() << endl; cout.flush();
	if (argv[2][0] == '1') {
		cout << "\nFrequent item sets: " << endl; cout.flush();
		set<int*>::iterator it;
		for (it = frequentSets.begin(); it != frequentSets.end(); it++) {
			set<string> *q = dGPU->bitToRecordSet(*it);
			for (set<string>::iterator jt = q->begin(); jt != q->end(); jt++) {
				cout << *jt << " ";
			}
			double support = dGPU->supportRate(*it);
			cout << support << endl; cout.flush();
			delete q;

		}

		cout << "\nStrong rules: " << endl; cout.flush();
		for (it = frequentSets.begin(); it != frequentSets.end(); it++) {
			set<string> *q = dGPU->bitToRecordSet(*it);
			set<string> ls, rs;
			findRulesGPU(*q, q->begin(), ls, rs);
			delete q;
		}
	}
	cout << "\nTotal time: " << (GetTickCount64() - totaltick) / 1000.0 << endl; cout.flush();
	cout << "GPU time: " << gputick / 1000.0 << endl; cout.flush();
	return 0;
}


int main(int argc, char** argv) {
	cout << "\n\n-- GPU test --" << endl;
	runOnGPU(argc, argv);

	cout << "\n\n-- CPU test --" << endl;
	runOnCPU(argc, argv);
	
}