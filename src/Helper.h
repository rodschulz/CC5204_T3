/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class Helper
{
public:
	template<class T> static void printMatrix(const Mat &_matrix, const int _precision = 1, const string &_name = "")
	{
		string format = "%- 15." + to_string(_precision) + "f\t";

		printf("%s\n", _name.c_str());
		for (int i = 0; i < _matrix.rows; i++)
		{
			for (int j = 0; j < _matrix.cols; j++)
			{
				printf(format.c_str(), _matrix.at<T>(i, j));
			}
			printf("\n");
		}
	}

	static int getRandomNumber(const int _min, const int _max);
	static void getContentsList(const string &_folder, vector<string> &_fileList, const bool _appendToList = false);
	static void createImageSamples(const string &_inputFolder, const double _sampleSize, const long _seed = -1);
	static void getClassNames(const string &_inputFolder, vector<string> &_classNames);
	static void calculateImageDescriptors(const string &_imageLocation, Mat &_descriptors);
	static size_t calculateHash(const vector<string> &_imageLocationList, const int _clusterNumber);
	static void concatMats(vector<Mat> &_vec, Mat &_res);
	static bool fileExists(const char *_filename);

private:
	Helper();
	~Helper();
};

