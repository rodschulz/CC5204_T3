//
// Created by fran on 5/20/15.
//

#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "svm.h"

using namespace std;
using namespace cv;

svm::svm(){
}

svm::~svm(){
}

// Set up training data
CvSVM svm::svmTrain(const vector<Mat> _bows){

    int totalImages = 0;
    int totalFeatures = _bows[0].cols;
    int nClasses = _bows.size();

    for(int k = 0; k < nClasses; k++){
        totalImages += _bows[k].rows;
    }

    int labels [435] = {}; //TODO ojo con esto
    Mat trainingData(totalImages, totalFeatures, CV_32FC1);

    int t = 0;
    for(int i = 0; i < nClasses ; i++){
        Mat currClass = _bows[i];
        int rows = currClass.rows;
        for(int j = 0; j < rows; j++){
            //labels.push_back(i); // Add the "class label", same for all the class data
            labels[t] = i;
            currClass.row(j).copyTo(trainingData.row(t));
            t++;
        }
    }

    Mat labelsMat(435, 1, CV_32FC1, labels);

    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    // Finishing criteria
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Model SVM
    CvSVM SVM;
    SVM.train(trainingData, labelsMat, Mat(), Mat(), params);

    return SVM;
}