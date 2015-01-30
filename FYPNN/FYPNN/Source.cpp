// TrainNetwork.cpp : Defines the entry point for the console application.

#include "opencv2/opencv.hpp"    // opencv general include file
#include "opencv2/ml/ml.hpp"          // opencv machine learning include file
#include <stdio.h>
#include <fstream>
#include <windows.h>
#include  "dirent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;
/******************************************************************************/

#define TRAINING_SAMPLES 234//Number of samples in training dataset
#define ATTRIBUTES 400  //Number of pixels per sample.16X16
#define TEST_SAMPLES 10    //Number of samples in test dataset
#define CLASSES 8         //Number of distinct labels.
#define IMGDIM 400 
#define MULTY 1 
/********************************************************************************
This function will read the csv files(training and test dataset) and convert them
into two matrices. classes matrix have 10 columns, one column for each class label. If the label of nth row in data matrix
is, lets say 5 then the value of classes[n][5] = 1.
********************************************************************************/


void read_train_image_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples);
void read_test_image_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples);
void read_train_digit_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples);
void train();
void test();
void genataer_result(char *filename, int count);
vector<string> get_all_files_names_within_folder(string folder);
void setUpTestSet(char *filename, int total_samples);
void setUpTestSet(char *filename, int total_samples);


void setUpTraingSet(char *filename, int total_samples)
{

	string p = filename + string() +"//train.csv";
	FILE* outputfile = fopen( p.c_str() , "w" );

	DIR *dir;
	struct dirent *ent;
	vector<string> out;
	/* open directory stream */
	dir = opendir (filename);
	if (dir != NULL) {

		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//printf ("%s\n", ent->d_name);
			out.push_back(ent->d_name);
		}
		closedir (dir);

	}

	//read each row of the csv file
	int count =0;

	for(int row = 2; row < out.size(); row++)
	{

		string path = filename + out[row].substr(0,out[row].size()-3)+"txt";
		FILE* inputfile = fopen( path.c_str(), "r" );
		

		if(out[row].find("jpg")!=std::string::npos)
		{

			int val;

			fscanf(inputfile, "%i,", &val);

			fprintf(outputfile,"%i,",val);


			Mat Img = imread(filename+out[row]);
			
			cout<< row <<endl;
			
			cvtColor(Img,Img,CV_BGR2GRAY);

			GaussianBlur(Img,Img,Size(7,7),0,0);
			
			equalizeHist(Img,Img);
			resize(Img,Img, Size(IMGDIM,IMGDIM));

			cv::Mat w(1,ATTRIBUTES,CV_32F);
			cv::Mat dis(IMGDIM,IMGDIM,CV_32F);
			for (int i = 0; i < IMGDIM; i++)
			{
				for (int j = 0; j < IMGDIM; j++)
				{
					float x0 = IMGDIM/2;
					float y0 = IMGDIM/2;
					float sigma =IMGDIM/4;
					float a = ( i - x0 ) * ( i - x0 )/(2*sigma*sigma);
					float b = ( j - x0 ) * ( j - x0 )/(2*sigma*sigma);

					//w.at<float>(0,i*IMGDIM+j) = exp(-a-b);

					dis.at<float>(i,j) =  exp(-a-b);
					Img.at<uchar>(i,j) *= exp(-a-b);
				}
			}
			imshow("w",dis);
			imshow("dis",Img);


			waitKey(100);
			for (int i = 0; i < Img.rows; i++)
			{
				for (int j = 0; j < Img.cols; j++)
				{
					fprintf(outputfile,"%i,",Img.at<uchar>(i,j));
				}
			}
			
			fprintf(outputfile,"\n");
			count++;
			fclose(inputfile);
			if(count>=total_samples)
			{
				break;
			}
		}
	}
	fclose(outputfile);

}


void setUpTestSet(char *filename, int total_samples)
{

	string p = filename + string() +"//test.csv";
	FILE* outputfile = fopen( p.c_str() , "w" );

	DIR *dir;
	struct dirent *ent;
	vector<string> out;
	/* open directory stream */
	dir = opendir (filename);
	if (dir != NULL) {

		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//printf ("%s\n", ent->d_name);
			out.push_back(ent->d_name);
		}
		closedir (dir);

	}

	//read each row of the csv file
	int count =0;

	for(int row = 2; row < out.size(); row++)
	{

		string path = filename + out[row].substr(0,out[row].size()-3)+"txt";

		if(out[row].find("jpg")!=std::string::npos)
		{
			// cout<< filename+out[row] <<endl;

			Mat Img = imread(filename+out[row]);
			cout<< row <<endl;
			resize(Img,Img, Size(IMGDIM,IMGDIM));
			cvtColor(Img,Img,CV_BGR2GRAY);
			imshow("out1",Img);
			waitKey(100);
			for (int i = 0; i < Img.rows; i++)
			{
				for (int j = 0; j < Img.cols; j++)
				{
					fprintf(outputfile,"%i,",Img.at<uchar>(i,j));
				}
			}
			
			fprintf(outputfile,"\n");
			count++;
			if(count>=total_samples)
			{
				break;
			}
		}
	}
	fclose(outputfile);

}
void read_dataset(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{

	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen( filename, "r" );

	//read each row of the csv file
	for(int row = 0; row < total_samples; row++)
	{
		//for each attribute in the row
		for(int col = 0; col <=ATTRIBUTES; col++)
		{
			//if its the pixel value.
			if (col < ATTRIBUTES){

				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row,col) = pixelvalue;

			}//if its the label
			else if (col == ATTRIBUTES){
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row,label) = 1.0;

			}
		}
	}

	fclose(inputfile);

}

/******************************************************************************/

int main( int argc, char** argv )
{
	setUpTraingSet("C:\\Users\\Chameera\\Desktop\\Bothale\\",TRAINING_SAMPLES);
	//setUpTestSet("C:\\Users\\Chameera\\Desktop\\Bothale\\",TEST_SAMPLES);
	train();
	//test();

}

void train(){

	
	//matrix to hold the training sample
	cv::Mat training_set(MULTY*TRAINING_SAMPLES,ATTRIBUTES,CV_32F);
	//matrix to hold the labels of each taining sample
	cv::Mat training_set_classifications(MULTY*TRAINING_SAMPLES, CLASSES, CV_32F);
	//matric to hold the test samples
	cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);
	//matrix to hold the test labels.
	cv::Mat test_set_classifications(TEST_SAMPLES,CLASSES,CV_32F);

	//
	cv::Mat classificationResult(1, CLASSES, CV_32F);




	//load the training and test data sets.
	read_train_digit_set("C:\\Users\\Chameera\\Desktop\\Bothale\\train.csv", training_set, training_set_classifications, TRAINING_SAMPLES);
	read_train_digit_set("C:\\Users\\Chameera\\Desktop\\Bothale\\train.csv", test_set, test_set_classifications, TEST_SAMPLES);

	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 256 input nodes
	// - 16 hidden nodes
	// - 10 output node, one for each class.

	cv::Mat layers(3,1,CV_32S);
	layers.at<int>(0,0) = ATTRIBUTES;//input layer
	layers.at<int>(1,0)=2000;//hidden layer
	//layers.at<int>(2,0)=1000;//hidden layer
	//layers.at<int>(3,0)=100;//hidden layer
	layers.at<int>(2,0) =CLASSES;//output layer

	//create the neural network.
	//for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
	CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);

	CvANN_MLP_TrainParams params(                                  

		// terminate the training after either 1000
		// iterations or a very small change in the
		// network wieghts below the specified value
		cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
		// use backpropogation for training
		CvANN_MLP_TrainParams::BACKPROP,
		// co-efficents for backpropogation training
		// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
		0.1,
		0.1);

	// train the neural network (using training data)

	printf( "\nUsing training dataset\n");


	int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
	printf( "Training iterations: %i\n\n", iterations);

	// Save the model generated into an xml file.
	CvFileStorage* storage = cvOpenFileStorage("C:\\Users\\Chameera\\Desktop\\Bothale\\param.xml", 0, CV_STORAGE_WRITE);
	nnetwork.write(storage,"DigitOCR");
	cvReleaseFileStorage(&storage);

	// Test the generated model with the test samples.
	cv::Mat test_sample;
	//count of correct classifications
	int correct_class = 0;
	//count of wrong classifications
	int wrong_class = 0;

	//classification matrix gives the count of classes to which the samples were classified.
	int classification_matrix[CLASSES][CLASSES]={{}};

	// for each sample in the test set.
	for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

		// extract the sample

		test_sample = test_set.row(tsample);

		//try to predict its class

		nnetwork.predict(test_sample, classificationResult);
		/*The classification result matrix holds weightage  of each class.
		we take the class with the highest weightage as the resultant class */

		// find the class with maximum weightage.
		int maxIndex = 0;
		float value=0.0f;
		float maxValue=classificationResult.at<float>(0,0);
		for(int index=1;index<CLASSES;index++)
		{   value = classificationResult.at<float>(0,index);
		if(value>maxValue)
		{   maxValue = value;
		maxIndex=index;

		}
		}

		printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);

		//Now compare the predicted class to the actural class. if the prediction is correct then\
		//test_set_classifications[tsample][ maxIndex] should be 1.
		//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex)!=1.0f)
		{
			// if they differ more than floating point error => wrong class

			wrong_class++;

			//find the actual label 'class_index'
			for(int class_index=0;class_index<CLASSES;class_index++)
			{
				if(test_set_classifications.at<float>(tsample, class_index)==1.0f)
				{

					classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
					break;
				}
			}

		} else {

			// otherwise correct

			correct_class++;
			classification_matrix[maxIndex][maxIndex]++;
		}
	}

	printf( "\nResults on the testing dataset\n"
		"\tCorrect classification: %d (%g%%)\n"
		"\tWrong classifications: %d (%g%%)\n", 
		correct_class, (double) correct_class*100/TEST_SAMPLES,
		wrong_class, (double) wrong_class*100/TEST_SAMPLES);
	cout<<"   ";
	for (int i = 0; i < CLASSES; i++)
	{
		cout<< i<<"\t";
	}
	cout<<"\n";
	for(int row=0;row<CLASSES;row++)
	{cout<<row<<"  ";
	for(int col=0;col<CLASSES;col++)
	{
		cout<<classification_matrix[row][col]<<"\t";
	}
	cout<<"\n";
	}

	getchar();
}

void test()
{
	genataer_result("C:\\Users\\Chameera\\Desktop\\Bothale\\test.csv",10);
}


void read_train_image_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{
	DIR *dir;
	struct dirent *ent;
	vector<string> out;
	/* open directory stream */
	dir = opendir (filename);
	if (dir != NULL) {

		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//printf ("%s\n", ent->d_name);
			out.push_back(ent->d_name);
		}
		closedir (dir);

	}

	//read each row of the csv file
	int count =0;

	int class_count[8] = {0,0,0,0,0,0,0,0};

	for(int row = 2; row < out.size(); row++)
	{
		if(out[row].find("jpg")!=std::string::npos)
		{
			// cout<< filename+out[row] <<endl;

			Mat Img = imread(filename+out[row]);
			cout<< row <<endl;



			resize(Img,Img, Size(IMGDIM,IMGDIM));


			cvtColor(Img,Img,CV_BGR2GRAY);



			imshow("out1",Img);
			waitKey(100);
			for (int i = 0; i < Img.rows; i++)
			{
				for (int j = 0; j < Img.cols; j++)
				{
					data.at<float>(MULTY*count,i*IMGDIM+j) = Img.at<uchar>(i,j);
					//data.at<float>(MULTY*count+1,i*IMGDIM+j) = Img.at<uchar>(i,Img.cols-j-1);
					//data.at<float>(MULTY*count+2,i*IMGDIM+j) = Img.at<uchar>(Img.rows - i -1,j);
				}
			}
			string path = filename + out[row].substr(0,out[row].size()-3)+"txt";
			FILE* inputfile = fopen( path.c_str(), "r" );
			int val;
			int val_x;
			int val_y;
			fscanf(inputfile, "%i,", &val);
			classes.at<float>(MULTY * count,val-1) = 1.0;

			switch(val)
			{
			case 1:
				val_x = 4;
				val_y = 8;
				break;
			case 2:
				val_x = 3;
				val_y = 7;
				break;
			case 3:
				val_x = 2;
				val_y = 6;
				break;
			case 4:
				val_x = 1;
				val_y = 5;
				break;
			case 5:
				val_x = 8;
				val_y = 4;
				break;
			case 6:
				val_x = 7;
				val_y = 3;
				break;
			case 7:
				val_x = 6;
				val_y = 2;
				break;
			case 8:
				val_x = 5;
				val_y = 1;
				break;
			}

			//classes.at<float>(MULTY * count+1,val_y-1) = 1.0;
			//classes.at<float>(MULTY * count+1,val_x-1) = 1.0;

			class_count[val-1]++;
			//class_count[val_y-1]++;
			//class_count[val_x-1]++;

			count++;
			fclose(inputfile);
			if(count>=total_samples)
			{
				break;
			}
		}
	}
	for (int i = 0; i < 8; i++)
	{
		printf("Calss %d : %d\n",(i+1),class_count[i]);
	}
}

void read_test_image_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{
	DIR *dir;
	struct dirent *ent;
	vector<string> out;
	/* open directory stream */
	dir = opendir (filename);
	if (dir != NULL) {

		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//printf ("%s\n", ent->d_name);
			out.push_back(ent->d_name);
		}
		closedir (dir);

	}

	//read each row of the csv file
	int count =0;



	for(int row = 2; row < out.size(); row++)
	{
		if(out[row].find("jpg")!=std::string::npos)
		{
			// cout<< filename+out[row] <<endl;

			Mat Img = imread(filename+out[row]);
			/*
			vector<Mat> channels; 
			Mat img_hist_equalized;

			cvtColor(Img, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

			split(img_hist_equalized,channels); //split the image into channels

			equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

			merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

			cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

			Img  = img_hist_equalized;
			*/
			cout<< row <<endl;
			cvtColor(Img,Img,CV_BGR2GRAY);

			//	resize(Img,Img, Size(),0.25,0.25);


			//GaussianBlur(Img,Img,Size(7,7),0,0);
			equalizeHist(Img,Img);

			threshold(Img,Img,10,255,CV_THRESH_BINARY_INV);
			//	imshow("asdddd",Img);

			//adaptiveThreshold(Img, Img,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,15,2);
			//Canny(Img,Img,50,150,3);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			/// Find contours
			//findContours( Img, contours, hierarchy, CV_RETR_EXTERNAL  , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

			/// Draw contours
			//Mat drawing = Mat::zeros( Img.size(), CV_8UC3 );
			//for( int i = 0; i< contours.size(); i++ )
			//{
			//	double area = contourArea(contours[i]);

			//	if( area > 300 )
			//	{
			//		Scalar color = Scalar( 200,200,200);
			//		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
			////	}
			//}

			//approxPolyDP(
			//cvtColor(drawing,drawing,CV_BGR2GRAY);
			//GaussianBlur(Img,Img,Size(9,9),0,0);
			//GaussianBlur(Img,Img,Size(9,9),0,0);
			//adaptiveThreshold(drawing, drawing,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,7,5);


			//vector<vector<Point> > newcontours;
			//vector<Vec4i> newhierarchy;

			/// Find contours
			//findContours( Img, newcontours, newhierarchy, CV_RETR_EXTERNAL  , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

			/// Draw contours
			//Mat newdrawing = Mat::zeros( Img.size(), CV_8UC3 );
			//for( int i = 0; i< newcontours.size(); i++ )
			//{
			//double area = contourArea(newcontours[i]);

			//if( area > 500 )
			//{
			//Scalar color = Scalar( 200,200,200);
			//drawContours( newdrawing, newcontours, i, color, 2, 8, hierarchy, 0, Point() );
			//}
			//}



			//imshow("out11",newdrawing);
			//Img = newdrawing;
			//imshow("out10",Img);
			resize(Img,Img, Size(IMGDIM,IMGDIM));
			//blur( Img, Img, Size(3,3) );


			//imshow("asdsdf",Img);

			//imshow("asd",Img);
			//GaussianBlur(Img,Img,Size(3,3),0,0);
			//

			imshow("out1",Img);


			cv::Mat w(1,ATTRIBUTES,CV_32F);
			cv::Mat dis(IMGDIM,IMGDIM,CV_32F);
			for (int i = 0; i < IMGDIM; i++)
			{
				for (int j = 0; j < IMGDIM; j++)
				{
					float x0 = IMGDIM/2;
					float y0 = IMGDIM/2;
					float sigma =IMGDIM/4;
					float a = ( i - x0 ) * ( i - x0 )/(2*sigma*sigma);
					float b = ( j - x0 ) * ( j - x0 )/(2*sigma*sigma);

					w.at<float>(0,i*IMGDIM+j) = exp(-a-b);

					dis.at<float>(i,j) =  exp(-a-b);
				}
			}
			imshow("dis",dis);


			waitKey(100);
			for (int i = 0; i < Img.rows; i++)
			{
				for (int j = 0; j < Img.cols; j++)
				{
					data.at<float>(count,i*IMGDIM+j) = (((float)Img.at<uchar>(i,j))/255 )* w.at<float>(0,i*IMGDIM+j);
					dis.at<float>(i,j) = (((float)Img.at<uchar>(i,j))/255 ) * w.at<float>(0,i*IMGDIM+j);
					//data.at<float>(count,i*IMGDIM+3*j) = Img.at<cv::Vec3b>(i,j)[0];
					//data.at<float>(count,i*IMGDIM+3*j+1) = Img.at<cv::Vec3b>(i,j)[1];
					//data.at<float>(count,i*IMGDIM+3*j+2) = Img.at<cv::Vec3b>(i,j)[2];

				}
			}

			imshow("dis1",dis);
			string path = filename + out[row].substr(0,out[row].size()-3)+"txt";
			FILE* inputfile = fopen( path.c_str(), "r" );
			int val;
			fscanf(inputfile, "%i,", &val);
			cout<<val<<endl;
			classes.at<float>(count,val-1) = 1.0;

			count++;
			fclose(inputfile);
			if(count>=total_samples)
			{
				break;
			}
		}
	}
}

void read_train_digit_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{
	string path = filename;
	FILE* inputfile = fopen( path.c_str(), "r" );
	int val;

	for (int i = 0; i < total_samples-1; i++)
	{

		fscanf(inputfile, "%i,", &val);
		classes.at<float>(i,val-1) = 1.0;

		for (int j = 0; j < ATTRIBUTES; j++)
		{
			fscanf(inputfile, "%i,", &val);
			data.at<float>(i,j) = val;
		}
	}	
	fclose(inputfile);
}

void genataer_result(char *filename, int count){
	//read the model from the XML file and create the neural network.
    CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( "F:\\Documents\\Projects\\Final Year Project\\NN\\NN_Train\\DS3\\Red_Bottle_Down\\param.xml", 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);
 
    //your code here
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel
    // ... data for the digit to be recognized
    // ...

	string path = filename;
	FILE* inputfile = fopen( path.c_str(), "r" );
	FILE* outputfile = fopen( "F:\\Documents\\Projects\\Final Year Project\\NN\\NN_Train\\DS3\\Red_Bottle_Down\\submit.csv", "w" );
	int val;

	std::fprintf(outputfile,"ImageId,Label\n");
	for (int i = 0; i < count; i++)
	{
		cv::Mat data(1,ATTRIBUTES,CV_32F);

		for (int j = 0; j < ATTRIBUTES; j++)
		{
			fscanf(inputfile, "%i,", &val);
			data.at<float>(0,j) = val;
		}


		int maxIndex = 0;
		cv::Mat classOut(1,CLASSES,CV_32F);
		//prediction
		nnetwork.predict(data, classOut);
		float value;
		float maxValue=classOut.at<float>(0,0);
		for(int index=1;index<CLASSES;index++)
		{   value = classOut.at<float>(0,index);
				if(value>maxValue)
				{   maxValue = value;
					maxIndex=index;
				}
		}
		std::fprintf(outputfile,"%i,%i\n",(i+1),maxIndex);
	}
	fclose(inputfile);
	fclose(outputfile);
    
    //maxIndex is the predicted class.
}


