//
// Created by fran on 5/20/15.
//

#ifndef CC5204_T3_SVM_H
#define CC5204_T3_SVM_H

#endif //CC5204_T3_SVM_H

using namespace std;
using namespace cv;

class svm
{
public:
    static void setUpTrainData(const vector<Mat> _bows, vector<Mat> &_toTrain, const int _labelsSize);
    static void loadSVMParams(CvSVMParams &_params, const string &_filename);

private:
    svm();
    ~svm();
};
