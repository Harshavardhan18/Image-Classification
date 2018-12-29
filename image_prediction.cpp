#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

HOGDescriptor hog(
        Size(256,256), //winSize
        Size(16,16), //blocksize
        Size(8,8), //blockStride,
        Size(8,8), //cellSize,
                18, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);

void computeHOG(vector<Mat> &inputdata, vector<vector<float> > &outputHOG) {
        for (int y = 0; y<inputdata.size(); y++) {
		vector<float> descriptors;
		hog.compute(inputdata[y], descriptors);
		outputHOG.push_back(descriptors);
	};
}

void ConvertVectortoMatrix(vector<vector<float> > &ipHOG, Mat & opMat)
{
	int descriptor_size = ipHOG[0].size();
	for (int i = 0; i<ipHOG.size(); i++) {
		for (int j = 0; j<descriptor_size; j++) {
			opMat.at<float>(i, j) = ipHOG[i][j];
		}
	}
}

void SVMPredict(Mat &testMat,Mat &res) {
    Ptr<SVM> svmNew = SVM::create();
    svmNew = StatModel::load<SVM> ("data.yml");
    svmNew->predict(testMat, res);
}

int main(int argc, char *argv[]) {
    
    if(argc<1) {
        exit(0);     
    } 
    
    int i = 1;
    vector<Mat> images;

    while(argv[i]) {
        Mat img = imread(argv[i],IMREAD_GRAYSCALE);
        i++;
        images.push_back(img);
    }
    
    vector<vector<float>> testHOG;
    
    computeHOG(images,testHOG);
    int descriptor_sz = testHOG[0].size();    
   
    Mat testMat(testHOG.size(), descriptor_sz, CV_32FC1);
    
    ConvertVectortoMatrix(testHOG, testMat);
    
    Mat res;
    
    SVMPredict(testMat,res);
    
    for(int i = 0;i<res.rows;i++)
    cout<<"Predicted class"<<res.at<float>(i,0)<<endl;
    return -1;
}

