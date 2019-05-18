#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <iostream>

using namespace std;

// Dataset
// Data is stored in row-set base
// NO labels are needed
class DataSet {
private:
	// Number of attributes in dataset
	int nAttr;

	
	// List of label (following the input order)
	vector<string> *label = NULL;

	
	// Map of data
	// (key, value) = (Attribute name, Set of value)
	map<string, set<string>*>* data;

	// Value count
	map<string, int> *vCount;
	
public:
	// Default constructor
	DataSet() {
		data = new map<string, set<string>*>();
		vCount = new map<string, int>();
	};

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

	// Add a value to an attribute
	//		@attr: attribute name (column)
	//		@fieldName: field name (row id)
	bool insert(string &key,string &value);

	// Check if a (key,value) exists in dataset
	//		@key: row name
	//		@value: value
	bool checkIfExist(string &key, string &value);

	// Check if a key exists in dataset
	//		@key: row name
	bool checkIfExist(string &key);

	// Remove a value from dataset
	//		@attr: attribute name (column)
	//		@fieldName: field name (row id)
	bool remove(string &key, string &value);

	// Remove a value from every set in dataset
	//		@value: value to remove
	bool removeValue(string &value);

	// Remove a key from dataset
	//		@value: value to remove
	bool removeKey(string &key);

	// Set label list
	//		@labelList: vector of labels
	void setLabelList(vector<string> &labelList);

	// Get label at index (1-index)
	//		@index: index of label to get
	string &getLabelAt(int index) {
		return label->at(index);
	}

	// Get number of key in dataset
	int keyCount();

	// Get number of value in a key
	// Return -1 if the key does not exist
	//		@key: key to count
	int valueCount(string &key);

	// Value count statistics
	map<string, int>* valueCount();
};


// Read data from csv file and return a DataSet
DataSet *readDataFromCSV(string filename);

// Read data from custom dat file and return a Dataset
DataSet *readDataFromCustomFile(string filename);
#endif