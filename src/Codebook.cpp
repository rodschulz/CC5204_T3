/**
 * Author: rodrigo
 * 2015
 */
#include "Codebook.h"
#include "Helper.h"
#include <fstream>

Codebook::Codebook(const int _clusterNumber)
{
	Mat labels = Mat();
	Mat centers = Mat();
	clusterNumber = _clusterNumber;
	dataHash = 0;
	indexBuilt = false;
}

Codebook::~Codebook()
{
}

ostream &operator<<(ostream &_stream, const Codebook &_codebook)
{
	int rows = _codebook.centers.rows;
	int cols = _codebook.centers.cols;
	int dataType = _codebook.centers.type();

	_stream << _codebook.dataHash << "\n";
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (dataType == CV_64FC1)
				_stream << _codebook.centers.at<double>(i, j) << " ";
			else if (dataType == CV_32FC1)
				_stream << _codebook.centers.at<float>(i, j) << " ";
		}
		_stream << "\n";
	}

	return _stream;
}

void Codebook::calculateCodebook(const string &_dataLocation, const int _maxInterationNumber, const double _stopThreshold)
{
	vector<string> imageLocationList;
	Helper::getContentsList(_dataLocation, imageLocationList);

	Ptr<FeatureDetector> featureExtractor = FeatureDetector::create("HARRIS");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	vector<Mat> descriptors;
	descriptors.reserve(imageLocationList.size());

	string names = "";

	Mat samples;
	initModule_nonfree();
	for (string imageLocation : imageLocationList)
	{
		Mat image = imread(imageLocation, CV_LOAD_IMAGE_GRAYSCALE);

		vector<KeyPoint> keypoints;
		featureExtractor->detect(image, keypoints);
		descriptors.push_back(Mat());
		descriptorExtractor->compute(image, keypoints, descriptors.back());

		if (samples.rows == 0)
			descriptors.back().copyTo(samples);
		else
			vconcat(samples, descriptors.back(), samples);

		names += imageLocation.substr(imageLocation.find_last_of('/') + 1);
	}

	int attempts = 5;
	kmeans(samples, clusterNumber, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, _maxInterationNumber, _stopThreshold), attempts, KMEANS_PP_CENTERS, centers);

	// Hash of the files used for the codebook (just the names for now)
	hash<string> strHash;
	dataHash = strHash(names);
}

void Codebook::saveToFile(const string &_destinationFolder) const
{
	fstream cacheFile;
	cacheFile.open(_destinationFolder + to_string(dataHash) + ".dat", fstream::out);
	cacheFile << *this;
	cacheFile.close();
}

void Codebook::getBoW(const Mat &_descriptors, Mat &_BoW)
{
	if (!indexBuilt)
		buildIndex();

	_BoW = Mat::zeros(1, centers.rows, CV_32FC1);
	for (int i = 0; i < _descriptors.rows; i++)
	{
		Mat indices, distances, currentRow;
		_descriptors.row(i).copyTo(currentRow);
		index.knnSearch(currentRow, indices, distances, 1);

		int tt = indices.type();
		_BoW.at<float>(1, indices.at<int>(0, 0)) += 1;

		Helper::printMatrix<int>(_BoW, 1, "BoW");
	}
}

void Codebook::buildIndex()
{
	index = flann::Index(Mat(centers).reshape(1), indexParams);
	indexBuilt = true;
}

bool Codebook::loadCodebook(const string &_imageSampleLocation, vector<Codebook> &_codebooks)
{
	return false;
}
