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

/*
	Change the className label according to your data requirements. 
*/
int className[3] = {0,1,2};

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
	}
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

void getSVMParams(SVM *svm)
{
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}

 
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {
 
	for (int i = 0; i<testResponse.rows; i++)
	{
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
		}
	}
        cout<<count<<endl;
	accuracy = (count / testResponse.rows) * 100;
}

void SVMtrain(Mat &trainMat, vector<int> &trainLabels, Mat &testResponse, Mat &testMat) {
        cout<<"Defining SVM"<<endl;
	Ptr<SVM> svm = SVM::create();
        svm->setC(1.1);
        svm->setCoef0(1);
        svm->setDegree(1);	
	svm->setGamma(0.75);
        svm->setNu(0.8);
	svm->setKernel(SVM::LINEAR);
	svm->setType(SVM::NU_SVC);
        svm->setTermCriteria(TermCriteria( TermCriteria::MAX_ITER, 10000, 1e-9));
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
        svm->train(td);
        
	svm->save("data.yml");
        svm->predict(testMat, testResponse);
	getSVMParams(svm); 
}

int main(int argc, char *argv[]) {
    
    vector<String> fileName,testImgName;
    vector<int> labels,testlabels;
    struct dirent *entry;
    
    vector<Mat> traindata,testdata;
    vector<int> trainlabel,testlabel;
/*
	The below code i=0 to 3 and i = 4 to 6 have been hard coded based on my dataset for training and testing. Please change it as per your training requirement.
*/
    cout<<"Fetching training images"<<endl;
    for(int i=0; i<3;i++) {
        int count = 0;
        DIR *dir = opendir(argv[i+1]);
        String path = argv[i+1];
        if (dir == NULL)
            exit(0);
        while (auto entry = readdir(dir)) {
            if (!entry->d_name || entry->d_name[0] == '.')
                continue;
            fileName.push_back(path+entry->d_name);
            labels.push_back(className[i]);
        }
        closedir(dir);
    }
    
    cout<<"Fetching test images"<<endl;
    for(int i=4; i<=6;i++) {
        int count = 0;
        DIR *dir = opendir(argv[i]);
        String path = argv[i];
        if (dir == NULL)
            exit(0);
        while (auto entry = readdir(dir)) {
            if (!entry->d_name || entry->d_name[0] == '.')
                continue;
            testImgName.push_back(path+entry->d_name);
            testlabels.push_back(className[i-4]);
        }
        closedir(dir);
    }

    cout<<"Training Image Count:"<<fileName.size()<<"\t"<<"Training Label Count:"<<labels.size()<<endl;
    cout<<"Test Image Count:"<<testImgName.size()<<"\t\t"<<"Test Label Count:"<<testlabels.size()<<endl;
    
    
    for (int i=0; i<testImgName.size(); i++) {
        Mat img = imread(testImgName[i],IMREAD_GRAYSCALE);    
        testdata.push_back(img);
        testlabel.push_back(testlabels[i]);
    }

    for (int i=0; i<fileName.size(); i++) {
        Mat img = imread(fileName[i],IMREAD_GRAYSCALE);       
        traindata.push_back(img);
        trainlabel.push_back(labels[i]);
    }
    
    vector<vector<float>> trainHOG;
    vector<vector<float>> testHOG;
    
    computeHOG(traindata,trainHOG);
    computeHOG(testdata,testHOG);
    

    int descriptor_sz = trainHOG[0].size();
    
    
    Mat trainM(trainHOG.size(), descriptor_sz,CV_32FC1);
    Mat testMat(testHOG.size(), descriptor_sz, CV_32FC1);
  

    ConvertVectortoMatrix(trainHOG, trainM);
    ConvertVectortoMatrix(testHOG, testMat);
  
    Mat testR;

    cout<<"Training the SVM"<<endl;
    SVMtrain(trainM, trainlabel,testR,testMat); 

    float count = 0;
    float accuracy = 0;
   
    cout<<"Evaluating the SVM"<<endl;
    
    SVMevaluate(testR, count, accuracy, testlabel);
 
    cout << "The accuracy of the model: " << accuracy << endl;
    return -1;
}
