#include "DataStructures.h"
#include <fstream>
#include <sstream>

DataSet::DataSet(vector<string>& label) {
	this->label = &label;
	this->data = new map<string, map<string, string>*>;
	for (int i = 0; i < (int)label.size(); i++) {
		data->insert({ label[i], new map<string,string>() });
	}
};

void DataSet::print() {
	cout << "Attributes set: ";
	for (map<string, map<string, string>*>::iterator i = data->begin(); i != data->end(); i++) {
		cout << i->first << " ";
	}
	cout << endl << endl;

	for (map<string, map<string, string>*>::iterator i = data->begin(); i != data->end(); i++) {
		cout << "Value in attribute \"" << i->first << "\": ";
		map<string, string>* currentAttr = i->second;
		for (map<string, string>::iterator j = currentAttr->begin(); j != currentAttr->end(); j++) {
			cout << "\"" << j->first << "\":\"" << j->second << "\" ";
		}
		cout << endl;
	}
}

bool DataSet::checkAttributeExist(string attr) {
	map<string, map<string, string>*>::iterator it = data->find(attr);
	return (it != data->end());
}

bool DataSet::insert(string attr, string fieldName, string value) {
	if (!checkAttributeExist(attr)) {
		cout << "ERROR: while adding value(s) to dataset: Attribute does not exist" << endl;
	}
	data->at(attr)->insert({ fieldName,value });
	return true;
}




DataSet *readDataFromCSV(string filename) {
	// Data format:
	// First row: "name", "attr1", "attr2",...
	// Each row: "name", "value of att1", "value of att2",...
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
	DataSet *d = new DataSet(*attrSet);

	// Get each line and parse its data
	while (getline(ifStream, line)) {
		if (line == "") continue;
		stringstream ss(line);

		// Get row name - the first value
		char c;
		string rowName = "";

		while (ss >> c) {
			rowName += c;
			if (ss.peek() == ',') {
				ss.ignore();
				break;
			}
		}
		d->insert(attrSet->at(0), rowName, rowName);
		// Now get these value
		int attrIndex = 1;
		string currentValue = "";
		while (ss >> c) {
			if (c == ',') {
				if (currentValue != "")
					d->insert(attrSet->at(attrIndex), rowName, currentValue);

				attrIndex++;
				currentValue = "";
			}
			else {
				currentValue += c;
			}
		}
		if (currentValue != "")  // Final value does not end up with ','
			d->insert(attrSet->at(attrIndex), rowName, currentValue);
	}

	return d;

}