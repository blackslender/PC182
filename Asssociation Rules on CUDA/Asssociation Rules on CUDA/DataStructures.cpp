#include "DataStructures.h"
#include <fstream>
#include <sstream>


void DataSet::print() {
	cout << "\nPrinting dataset: " << endl;
	for (map<string, set<string>*>::iterator i = data->begin(); i != data->end(); i++) {
		cout << "Key: \"" << i->first << "\"" << endl;
		set<string>* currentKey = i->second;
		for (set<string>::iterator j = currentKey->begin(); j != currentKey->end(); j++) {
			cout << "\"" << *j << "\" ";
		}
		cout << endl << endl;
	}
}

bool DataSet::insert(string &key, string &value) {
	if (!checkIfExist(key)) data->insert({ key, new set<string>() });
	data->at(key)->insert(value);
	return true;
}

bool DataSet::checkIfExist(string &key, string &value) {
	if (data->find(key) == data->end()) return false;
	if (data->find(key)->second->count(value) == 0) return false;
	return true;

}

bool DataSet::checkIfExist(string &key) {
	if (data->find(key) == data->end()) return false;
	return true;
}

bool DataSet::remove(string &key, string &value) {
	if (!checkIfExist(key, value)) {
		cout << "ERROR: while removing value(s) from dataset: key or value does not exist" << endl;
		return false;
	}
	data->find(key)->second->erase(value);
	return true;
};

bool DataSet::removeValue(string &value) {
	map<string, set<string>*>::iterator it;
	for (it = data->begin(); it != data->end(); it++) {
		set<string>* currentSet = it->second;
		currentSet->erase(value);
	}
	return true;
}

bool DataSet::removeKey(string &key) {
	if (!checkIfExist(key)) {
		cout << "ERROR: while removing key from dataset: key does not exist" << endl;
		return false;
	}
	data->find(key)->second->clear();
	data->erase(key);
	return true;
}

void DataSet::setLabelList(vector<string> &labelList) {
	this->label = &labelList;
}


// Value count statistics
map<string, int>* DataSet::valueCount() {
	map<string, set<string>*>::iterator iKey;
	for (iKey = data->begin(); iKey != data->end(); iKey++) {
		set<string>::iterator iValue;
		for (iValue = iKey->second->begin(); iValue != iKey->second->end(); iValue++) {
			if (vCount->find(*iValue) == vCount->end())
				vCount->insert({ *iValue,0 });
			(*vCount)[*iValue]++;

		}
	}
	return vCount;
}

// Data format: normal csv
// First row: "name", "attr1", "attr2",...
// Each row: "name", "value of att1", "value of att2",...
DataSet *readDataFromCSV(string filename) {
	string line;

	ifstream ifStream; ifStream.open(filename, ios::in);

	// Get first line to extract labels
	getline(ifStream, line);

	stringstream ss(line);
	char c;
	vector<string> *attrSet = new vector<string>();
	attrSet->push_back("");
	while (ss >> c) {
		attrSet->at(attrSet->size() - 1) += c;
		if (ss.peek() == ',') {
			ss.ignore();
			attrSet->push_back("");
		}
	}
	DataSet *d = new DataSet();
	d->setLabelList(*attrSet);

	// Get each line and parse its data
	while (getline(ifStream, line)) {
		if (line == "") continue;
		stringstream ss(line);

		// Get row name - the first value
		char c;
		string rowName = "_id_";

		while (ss >> c) {
			rowName += c;
			if (ss.peek() == ',') {
				ss.ignore();
				break;
			}
		}
		// Now get these value
		int attrIndex = 1;
		string currentValue = "";
		while (ss >> c) {
			if (c == ',') {
				if (currentValue == "Y" || currentValue == "y") { // We consider 'y', 'Y' as yes, others is considered as no
					d->insert(rowName, d->getLabelAt(attrIndex));
				}
				attrIndex++;
				currentValue = "";
			}
			else {
				currentValue += c;
			}
		}
		if (currentValue == "Y" || currentValue == "y") { // Final value
			d->insert(rowName, d->getLabelAt(attrIndex));
		}
	}

	return d;

}

// Data format: custom
// Each row: "value1" "value2" "value3" ...
DataSet *readDataFromCustomFile(string filename) {
	DataSet *d = new DataSet();
	string line;
	ifstream ifStream; ifStream.open(filename, ios::in);
	// Get each line and parse its data
	int lineID = 0;
	while (getline(ifStream, line)) {
		if (line == "") continue;
		stringstream ss(line);

		// Get row name - the first value
		char c;
		string rowName = "_id_" + to_string(++lineID);
		// Now get these value
		int attrIndex = 1;
		string currentValue = "";
		while (ss >> currentValue) {
			if (true) {
				if (currentValue != "" && currentValue != " ")
					d->insert(rowName, currentValue);
				currentValue = "";
			}
			else {
				currentValue += c;
			}
		}
		if (currentValue != "" && currentValue != " ")
			d->insert(rowName, currentValue);
	}
	return d;
}

int DataSet::keyCount() {
	return data->size();
}

int DataSet::valueCount(string &key) {
	if (!checkIfExist(key)) return -1;
	return data->find(key)->second->size();
}