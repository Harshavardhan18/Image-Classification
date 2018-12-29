#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;

int main(int argc,char** argv ) {
    if (argc < 1) {
        cout<<"Insufficient arguments"<<endl; 
        return -1;   
    }
    Mat image,rimg;
    int i = 1;
    while(argv[i]){
        image = imread(argv[i]);
        resize(image,rimg,Size(256,256),0,0,INTER_LINEAR);
        imwrite(argv[i],rimg);
        i++;
    }
    waitKey(0);
    return 0;
}
