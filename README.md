# Image-Classification

The image classification is designed in c++ and opencv - a image processing library in linux(ubuntu) platform.
Hi, I would like to provide a walkthrough of how i was able to implement the image classification using HOG and SVM. This may help someone who is getting started or anyone who wishes to implement the same.

Prerequisites:
1.  Install openncv.
2.  Basic Knowledge of c++.
3.  Dataset.
4.  Knowledge of HOG and SVM working.

I assume you already have installed a linux platform like ubuntu, RedHat, linuxMint etc.

Install Opencv
    Follow the guide lines in the link provided for install opencv in ur linux system.
    http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/
  
Gather dataset
    There many website that provide a wide range of image dataset. I have used the Monkey Species from Kaggle.
    Some of website are:
    * Kaggle
    * Google
    
Once the opencv has been installed and the dataset for which the model is to be trained has been gathered.
Prepare Training dataset:
*Create separate folder training dataset -> subfolder for each class 
*resize the dataset to have the same dimension (256 x 256)
	
Prepare Test dataset
*Create separate folder test dataset
*resize the dataset to have the dimension as in training process

Initially load the training set into a vector array.
Read the training set in opencv::Mat(GREY_SCALE). Store the Mat data in a vector with associated the class labels in vector.

Compute HOG

Create a SVM 
	use opencv library to define SVM
        opencv uses one-vs-one classification: given n classes creates n(n-1)/2 classifiers
        assign reqired parametes for training the svm	

Save and load the svm
Test the svm on testset for accuracy


To compile and execute the image_classification_training.cpp
* g++ -std=c++11 image_classification_training.cpp -o output `pkg-config --cflags --libs opencv`
* ./output (include command line arg if ur providing the location of training and test dataset)

Once the training has been completed a .yml file is created by the SVM. This file can later be used to predict the image instead of training the model each time.

To compile and execute image_prediction.cpp
* g++ -std=c++11 image_predictiong.cpp -o predict `pkg-config --cflags --libs opencv`
* ./predict (location of image data to predict)

Fell free to tune the SVM parameters referenced by the documentation(https://docs.opencv.org/3.2.0/d1/d2d/classcv_1_1ml_1_1SVM.html) for various kernal types, SVM Types and various parameter.

For more information on
* How HOG works?
  * https://www.learnopencv.com/histogram-of-oriented-gradients/
  * http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/

* How SVM works?
  * https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
  
* Opencv documentation
  * https://docs.opencv.org/3.2.0/
