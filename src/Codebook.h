/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

class Codebook
{
public:
	Codebook(const int _clusterNumber);
	~Codebook();

	friend ostream &operator<<(ostream &_stream, const Codebook &_codebook);

	void calculateCodebook(const string &_dataLocation, const int _maxInterationNumber, const double _stopThreshold);
	void saveToFile(const string &_destinationFolder) const;
	void getBoW(const Mat &_descriptor, Mat &_BoW);

	static bool loadCodebook(const string &_imageSampleLocation, vector<Codebook> &_codebooks);

private:
	void buildIndex();

	Mat labels;
	Mat centers;
	int clusterNumber;
	size_t dataHash;

	bool indexBuilt;
	flann::KDTreeIndexParams indexParams;
	flann::Index index;
};

