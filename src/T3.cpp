#include <stdlib.h>
#include <iostream>
#include <string>
#include "Helper.h"
#include "Codebook.h"
#include "Config.h"
#include "svm.h"

using namespace std;

void calculateBoWs(const string &_inputFolder, const vector<string> &_classNames, const string &_set, vector<Codebook> &_codebooks, vector<Mat> &_BoWs)
{
	for (size_t i = 0; i < _classNames.size(); i++)
	{
		string className = _classNames[i];
		cout << "Calculating BoW for class " << className << " using set " << _set << "\n";

		vector<string> imageList;
		Helper::getContentsList(_inputFolder + className + "/" + className + "_" + _set + "/", imageList);
		Mat currentClassBoW = Mat::zeros(imageList.size(), _codebooks[i].getClusterNumber(), CV_32FC1);
		_BoWs.push_back(currentClassBoW);

		int j = 0;
		Mat documentCounter = Mat::zeros(1, _codebooks[i].getClusterNumber(), CV_32FC1);

		cout << "Calculating frequencies\n";
		for (string imageLocation : imageList)
		{
			Mat descriptors;
			Helper::calculateImageDescriptors(imageLocation, descriptors);
			Mat row = currentClassBoW.row(j++);
			_codebooks[i].getBoWTF(descriptors, row);

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

		//Helper::printMatrix<float>(documentCounter, 3);
		//Helper::printMatrix<float>(currentClassBoW, 3);
	}
}

int main(int _nargs, char ** _vargs)
{
	if (_nargs < 2)
		cout << "Not enough arguments\n";

	string inputFolder = _vargs[1];
	cout << inputFolder << endl;
	Config::load("../config/config");

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
			codebooks.back().saveToFile("../cache/");
		}
		else
			cout << "Codebook for class '" << className << "' read from cache\n";
	}

	// Calculate the BoW for each image in each set
	vector<Mat> trainBoWs, validationBoWs, testBoWs;
	calculateBoWs(inputFolder, classNames, "train", codebooks, trainBoWs);
	calculateBoWs(inputFolder, classNames, "val", codebooks, validationBoWs);
	calculateBoWs(inputFolder, classNames, "test", codebooks, testBoWs);

	// Classification part
	svm::svmTrain(trainBoWs);
	//ACA EL SVM

	//Aca una iteracion para usar CvSVM::Predict con cada imagen de test

	cout << "Finished\n";
	return EXIT_SUCCESS;
}
