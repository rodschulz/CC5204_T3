/**
 * Author: rodrigo
 * 2015
 */
#include "Codebook.h"
#include "Helper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>

Codebook::Codebook(const int _clusterNumber)
{
	centers = Mat::zeros(1, 1, CV_32FC1);
	clusterNumber = _clusterNumber;
	dataHash = 0;
}

Codebook::Codebook(const Codebook &_other)
{
	centers = _other.centers.clone();
	clusterNumber = _other.clusterNumber;
	dataHash = _other.dataHash;
	index = _other.index;
}

Codebook::Codebook()
{
	centers = Mat::zeros(1, 1, CV_32FC1);
	clusterNumber = 1;
	dataHash = 0;
}

Codebook::~Codebook()
{
}

Codebook &Codebook::operator=(const Codebook &_other)
{
	if (this != &_other)
	{
		centers = _other.centers.clone();
		clusterNumber = _other.clusterNumber;
		dataHash = _other.dataHash;
		index = _other.index;
	}

	return *this;
}

ostream &operator<<(ostream &_stream, const Codebook &_codebook)
{
	int rows = _codebook.centers.rows;
	int cols = _codebook.centers.cols;
	int dataType = _codebook.centers.type();

	_stream << _codebook.dataHash << " " << rows << "\n";
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

	vector<Mat> descriptors;
	descriptors.reserve(imageLocationList.size());

	string names = "";

	Mat samples;
	for (string imageLocation : imageLocationList)
	{
		// Calculate image's descriptors
		descriptors.push_back(Mat());
		Helper::calculateImageDescriptors(imageLocation, descriptors.back());

		if (samples.rows == 0)
			descriptors.back().copyTo(samples);
		else
			vconcat(samples, descriptors.back(), samples);

		// Concatenate names to create a hash to identify the sample set
		names += imageLocation.substr(imageLocation.find_last_of('/') + 1);
	}

	int attempts = 5;
	Mat labels;
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

void Codebook::getBoWTF(const Mat &_descriptors, Mat &_BoW)
{
	static bool indexBuilt = false;
	if (!indexBuilt)
	{
		index.build(centers, flann::KDTreeIndexParams(4));
		indexBuilt = true;
	}

	if (_descriptors.rows > 0)
	{
		// Get frequencies of each word
		_BoW = Mat::zeros(1, centers.rows, CV_32SC1);
		for (int i = 0; i < _descriptors.rows; i++)
		{
			Mat indices, distances, currentRow;
			_descriptors.row(i).copyTo(currentRow);

			index.knnSearch(currentRow, indices, distances, 1);
			_BoW.at<int>(0, indices.at<int>(0, 0)) += 1;
		}

		Helper::printMatrix<int>(_BoW, 1);

		// Normalize using the total of word to get the TF
		_BoW *= (1 / _descriptors.rows);
		Helper::printMatrix<int>(_BoW, 1);
	}
}

bool Codebook::loadCodebook(const string &_imageSampleLocation, vector<Codebook> &_codebooks)
{
	// Calculate hash of data in sample
	vector<string> imageLocationList;
	Helper::getContentsList(_imageSampleLocation, imageLocationList);

	string names = "";
	for (string imageLocation : imageLocationList)
		names += imageLocation.substr(imageLocation.find_last_of('/') + 1);

	// Hash of the files used for the codebook (just the names for now)
	hash<string> strHash;
	size_t sampleHash = strHash(names);
	string filename = "./cache/" + to_string(sampleHash) + ".dat";

	bool codebookRead = false;
	string line;
	ifstream inputFile;
	inputFile.open(filename.c_str(), fstream::in);
	if (inputFile.is_open())
	{
		int rows = -1;
		int cols = -1;
		int i = 0;
		while (getline(inputFile, line))
		{
			vector<string> tokens;
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

			if (rows == -1)
			{
				if (tokens.size() != 3)
					break;

				rows = stoi(tokens[1]);
				cols = stoi(tokens[2]);
				_codebooks.push_back(Codebook(rows));
				_codebooks.back().dataHash = sampleHash;
				_codebooks.back().centers = Mat::zeros(rows, cols, CV_32FC1);
			}
			else
			{
				int j = 0;
				for (string value : tokens)
				{
					_codebooks.back().centers.at<float>(i, j++) = stof(value);
				}
				i++;
			}
		}
		inputFile.close();
		codebookRead = rows != -1 ? true : false;
	}

	return codebookRead;
}
