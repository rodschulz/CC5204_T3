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
    static vector<Mat> svmTrain(const vector<Mat> _bows);

private:
    svm();
    ~svm();
};
