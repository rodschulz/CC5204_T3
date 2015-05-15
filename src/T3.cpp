#include <stdlib.h>
#include <iostream>
#include <string>
#include "Helper.h"
#include "Codebook.h"
#include "Config.h"

using namespace std;

int main(int _nargs, char ** _vargs)
{
	if (_nargs < 2)
		cout << "Not enough arguments\n";

	string inputFolder = _vargs[1];
	Config::load("./config/config");

	// Create a new image sample
	if (Config::createImageSample())
	{
		cout << "Creating image sample\n";
		Helper::createImageSamples(inputFolder, Config::getSampleSize());
	}

	vector<Codebook> codebooks;

	// Generate or load the codebooks
	vector<string> classNames;
	Helper::getClassNames(inputFolder, classNames);
	for (string className : classNames)
	{
		if (!Codebook::loadCodebook(inputFolder + className + "/sample/", codebooks))
		{
			cout << "Codebook for class " << className << "not found in cache. Calculating new codebook\n";
			codebooks.push_back(Codebook(Config::getCodebookClustersNumber()));
			codebooks.back().calculateCodebook(inputFolder + className + "/sample/", 10000, 0.1);
			cout << "Saving codebook for class " << className << " to cache file\n";
			codebooks.back().saveToFile("./cache/");
		}
	}

	// Create Bow for every image in the train set
	for (size_t i = 0; i < classNames.size(); i++)
	{
		string className = classNames[i];

		vector<string> imageList;
		Helper::getContentsList(inputFolder + className + "/" + className + "_train/", imageList);
		for (string imageLocation : imageList)
		{
//			Mat kk;
//			codebook.getBoW(Mat(), kk);
		}
	}

	cout << "Finished\n";
	return EXIT_SUCCESS;
}
