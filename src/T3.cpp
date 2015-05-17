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
			cout << "Codebook for class '" << className << "' not found in cache. Calculating new codebook\n";
			codebooks.push_back(Codebook(Config::getCodebookClustersNumber()));
			codebooks.back().calculateCodebook(inputFolder + className + "/sample/", 10000, 0.1);
			cout << "Saving codebook for class '" << className << "' to cache file\n";
			codebooks.back().saveToFile("./cache/");
		}
		else
			cout << "Codebook for class '" << className << "' read from cache\n";
	}

	// Calculate the BoW for each image in the train set
	vector<Mat> BoWs;
	for (size_t i = 0; i < classNames.size(); i++)
	{
		string className = classNames[i];
		cout << "Calculating BoW for class " << className << "\n";

		vector<string> imageList;
		Helper::getContentsList(inputFolder + className + "/" + className + "_train/", imageList);
		Mat currentClassBoW = Mat::zeros(imageList.size(), codebooks[i].getClusterNumber(), CV_32FC1);
		BoWs.push_back(currentClassBoW);

		int j = 0;
		Mat documentCounter = Mat::zeros(1, codebooks[i].getClusterNumber(), CV_32FC1);

		cout << "Calculating frequencies\n";
		for (string imageLocation : imageList)
		{
			Mat descriptors;
			Helper::calculateImageDescriptors(imageLocation, descriptors);
			Mat row = currentClassBoW.row(j++);
			codebooks[i].getBoWTF(descriptors, row);

			for (int k = 0; k < currentClassBoW.cols; k++)
				documentCounter.at<float>(0, k) += (row.at<float>(0, k) > 0 ? 1 : 0);
		}

		cout << "Calculating tf-idf\n";
		// Calculate tf-idf logarithmic factor and then the tf-idf itself
		for (int k = 0; k < documentCounter.cols; k++)
			documentCounter.at<float>(0, k) = log((float) imageList.size() / documentCounter.at<float>(0, k));
		for (int p = 0; p < currentClassBoW.rows; p++)
		{
			for (int q = 0; q < currentClassBoW.cols; q++)
				currentClassBoW.at<float>(p, q) *= documentCounter.at<float>(0, q);
		}

		Helper::printMatrix<float>(documentCounter, 3);
		//Helper::printMatrix<float>(currentClassBoW, 3);
	}

	cout << "Finished\n";
	return EXIT_SUCCESS;
}
