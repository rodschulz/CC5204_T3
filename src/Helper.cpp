/**
 * Author: rodrigo
 * 2015
 */
#include "Helper.h"
#include <stdlib.h>
#include <dirent.h>
#include <cstring>
#include <iostream>

Helper::Helper()
{
}

Helper::~Helper()
{
}

int Helper::getRandomNumber(const int _min, const int _max)
{
	srand(rand());
	srand(rand());
	int number = _min + (rand() % (int) (_max - _min + 1));
	return number;
}

void Helper::getContentsList(const string &_folder, vector<string> &_fileList, const bool _appendToList)
{
	DIR *folder;
	struct dirent *epdf;

	if (!_appendToList)
		_fileList.clear();

	if ((folder = opendir(_folder.c_str())) != NULL)
	{
		while ((epdf = readdir(folder)) != NULL)
		{
			if (strcmp(epdf->d_name, ".") == 0 || strcmp(epdf->d_name, "..") == 0)
				continue;

			_fileList.push_back(_folder + epdf->d_name);
		}
		closedir(folder);
	}
	else
	{
		cout << "ERROR: can't open folder " << _folder << "\n";
	}
}

void Helper::createImageSamples(const string &_inputFolder, const double _sampleSize, const long _seed)
{
	vector<string> folderList;
	Helper::getContentsList(_inputFolder, folderList);

	if (_seed != -1)
		srand(_seed);
	else
	{
		srand(time(NULL));
		srand(rand());
		srand(rand());
	}

	string cmd;
	for (string classFolder : folderList)
	{
		string className = classFolder.substr(classFolder.find_last_of('/'));

		vector<string> classContents;
		Helper::getContentsList(classFolder + className + "_test/", classContents, true);
		Helper::getContentsList(classFolder + className + "_train/", classContents, true);
		Helper::getContentsList(classFolder + className + "_val/", classContents, true);

		string sampleFolder = classFolder + "/sample/";
		cmd = "rm -rf " + sampleFolder;
		system(cmd.c_str());
		cmd = "mkdir " + sampleFolder;
		system(cmd.c_str());

		vector<string> classSample;
		int sampleSize = classContents.size() * _sampleSize;
		for (int k = 0; k < sampleSize; k++)
		{
			int sampleIndex = (rand() % (int) classContents.size());
			string origin = *(classContents.begin() + sampleIndex);
			string destination = sampleFolder + origin.substr(origin.find_last_of('/') + 1);
			cmd = "cp " + origin + " " + destination;
			system(cmd.c_str());
			classContents.erase(classContents.begin() + sampleIndex);
		}
	}
}

void Helper::getClassNames(const string &_inputFolder, vector<string> &_classNames)
{
	_classNames.clear();
	Helper::getContentsList(_inputFolder, _classNames);
	for (size_t i = 0; i < _classNames.size(); i++)
		_classNames[i] = _classNames[i].substr(_classNames[i].find_last_of('/') + 1);
}