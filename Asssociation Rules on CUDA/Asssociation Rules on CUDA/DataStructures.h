#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <iostream>

using namespace std;

// Dataset
// Data is stored in column-base
class DataSet {
private:
	// List of label (following the input order)
	vector<string> *label = NULL;

	// Number of attributes in dataset
	int nAttr;

	// Map of data
	// (key, value) = (Attribute name, Set of value)
	map<string, map<string,string>*>* data = NULL;

	// Default constructor is disabled
	DataSet() {};
public:

	// Inital dataset constructor
	//		@label: vector of attribute names
	DataSet(vector<string>& label);

	// Destructor
	~DataSet() {
		if (label != NULL) {
			label->clear();
			delete label;
			label = NULL;
		}
		if (data != NULL) {
			data->clear();
			delete data;
			data = NULL;
		}
	}

	// Print the dataset to console for debugging
	void print();

	// Check if a attribute is in dataset
	bool checkAttributeExist(string attr);

	// Add a value to an attribute
	//		@attr: attribute name (column)
	//		@fieldName: field name (row id)
	bool insert(string attr, string fieldName,string value);


};


// Read data from csv file and return a DataSet
DataSet *readDataFromCSV(string filename);

#endif