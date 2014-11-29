/**
 * eye-tracking.cpp:
 * Eye detection and tracking with OpenCV
 *
 * This program tries to detect and tracking the user's eye with webcam.
 * At startup, the program performs face detection followed by eye detection
 * using OpenCV's built-in Haar cascade classifier. If the user's eye detected
 * successfully, an eye template is extracted. This template will be used in
 * the subsequent template matching for tracking the eye.
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <cstdio>
#include "constants.h"
#include <queue>
#include <math.h>
#include <fstream>

//#include "findEyeCenter.h"

#pragma once

#include "cv.h"
#include "highgui.h"
#include "math.h"
#include "list"

using namespace std;
using namespace cv;
/**
 * Function to detect human face and the eyes from an image.
 *
 * @param  im    The source image
 * @param  tpl   Will be filled with the eye template, if detection success.
 * @param  rect  Will be filled with the bounding box of the eye
 * @return zero=failed, nonzero=success
 */

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
int f = 0;
Mat eye_left_tpl, eye_right_tpl;
Rect rect1, rect2;

ofstream fout("output.txt");
double rxc = 0, rx1 = 0, rx2 = 0, rx3 = 0, rx4 = 0;
double ryc = 0, ry1 = 0, ry2 = 0, ry3 = 0, ry4 = 0;
int countc = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0;

class voteNode {
public:
	int x;
	int y;
};

void Hough(cv::Mat imge, double* circle)
{
    equalizeHist( imge, imge );
    IplImage tmp = imge;
    IplImage* img = &tmp;

	uchar* pxl_ptr;
	CvPixelPosition8u pos_src;
	//	Now we are pointing to the first pixel
	CV_INIT_PIXEL_POS(pos_src, (unsigned char *) img->imageData, img->widthStep, cvGetSize(img),0,0,img->origin);

	//	Define search space for radius
	const int search_xmin = 0;
	const int search_ymin = 0;
	const int search_xmax = img->width;//(img->width*7)/8;
	const int search_ymax = img->height;//(img->height*3)/4;

	const int x_max = img->width;
	const int y_max = img->height;
	const int r_max = min(img->width, img->height)/3;


	//	Our list to store radius information for each img pixel
	list<voteNode>*** P = new list<voteNode>** [x_max];

	int i,j;
	for (i = 0; i < x_max; i++)
	{
		P[i] = new list<voteNode>*[y_max];
		for (j = 0; j < y_max; j++)
			P[i][j] = new list<voteNode>[r_max];
	}

	//	Start searching img for 255 pixels
	int r;
	voteNode temp;
	int x, y;

	for (i = search_xmin; i < search_xmax; i++)
	{
		for (j = search_ymin; j < search_ymax; j++)
		{
			for (x = x_max/6; x < x_max*5/6; x++)
			{
				for (y = y_max/6; y < y_max*5/6; y++)
				{
					pxl_ptr = CV_MOVE_TO(pos_src, x, y, 1); // Now we point to the ixjth pixel

					if (*pxl_ptr <= 17)	// whites are edges
					{
					    //std::cout << "factor = " << cvRound( sqrt( pow((i-x), 2.0) + pow((j-y), 2.0) ) ) << std::endl;
						r = (int) cvRound( sqrt( pow((i-x), 2.0) + pow((j-y), 2.0) ) );
						temp.x = x; temp.y = y;
						if (r<r_max)
							P[i][j][r].push_front(temp);
					}
				}
			}
		}
	}

	int k;
	int cx, cy;

	double max_r = 0;
	int max_radius_vote = 0;
	list<voteNode>::iterator votes;
	bool flag;

	for (i = search_xmin; i < search_xmax; i++)
	{
		for ( j = search_xmin; j < search_ymax; j++)
		{
			// find highest voted radius
			for (k = 0; k < r_max; k++)
			{
				if (P[i][j][k].size() > max_radius_vote)
				{
					max_radius_vote = P[i][j][k].size();
					max_r = k;
					cx = i; cy = j;
				}
			}
		}
	}

	//	Get the coordinates and radius information
	circle[0] = cx; circle[1] = cy; circle[2] = max_r;


	//cout << "x= " << circle[0] << " y = " << circle[1] << " r = " << circle[2] << endl;

	delete []P;
}


int detectAndDisplay(Mat& frame, string WindowName)
{


    Mat frame_gray = frame;
   /// cvtColor( frame, frame_gray, CV_BGR2GRAY );
    //cout << "in detect \n";
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> eyes;
        //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( frame_gray, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    //cout << "eye = " << eyes.size() << endl;
        Rect swp;
    if (eyes.size() == 2) {
        if (rect1.x > rect2.x) {
                swp = rect1;
                rect1 = rect2;
                rect2 = swp;
        }
        rect1 = eyes[0];
        eye_left_tpl = frame_gray(rect1);
        rect2 = eyes[1];
        eye_right_tpl = frame_gray(rect2);
        f = 1;

        cv::namedWindow("left");
        imshow( "left", eye_left_tpl );

        cv::namedWindow("right");
        imshow( "right", eye_right_tpl );

    }
    //cout << "yes";
//-- Show what you got
    cv::namedWindow(WindowName);
    imshow( WindowName, frame );
    return f;
}

void findEyes(cv::Mat face) {

  //-- Find eye regions and draw them
  int eye_region_width = face.rows * (kEyePercentWidth/100.0);
  int eye_region_height = face.rows * (kEyePercentHeight/100.0);
  int eye_region_top = face.cols * (kEyePercentTop/100.0);

  //cv::Rect EyeRegion(0, eye_region_top, face.rows, eye_region_height);
  cv::Rect EyeRegion(face.rows * (.13), eye_region_top + eye_region_top / 4, face.rows / 2 - face.rows * (.13), eye_region_height - eye_region_top / 2);
  cv::Mat publish = face(EyeRegion);
  double center[3];

  Hough(publish, center);
  //cout << "x = " << center[0] << " " << "y = " << center[1] << endl;
  circle(publish, cv::Point(center[0], center[1]), center[2], 255, 1, 8, 0);
  //detectAndDisplay(publish, "left1");

  cv::Rect EyeRegion1((face.rows / 2), eye_region_top + eye_region_top / 4, face.rows / 2 - face.rows * (.13), eye_region_height - eye_region_top / 2);
  cv::Mat publish1 = face(EyeRegion1);
  double center1[3];

  Hough(publish1, center1);
  //cout << "x = " << center[0] << " " << "y = " << center[1] << endl;
  circle(publish1, cv::Point(center1[0], center1[1]), center1[2], 255, 1, 8, 0);
  cv::namedWindow("11foo1");


  char character;
  cin >> character;
  fout << character << endl;
  if (character == 'c') {
    rxc += center1[0];
    ryc += center1[1];
    countc++;
  } else if (character == '1') {
    rx1 += center1[0];
    ry1 += center1[1];
    count1++;
  } else if (character == '2') {
    rx2 += center1[0];
    ry2 += center1[1];
    count2++;
  } else if (character == '3') {
    rx3 += center1[0];
    ry3 += center1[1];
    count3++;
  } else if (character == '4') {
    rx4 += center1[0];
    ry4 += center1[1];
    count4++;
  }

  fout << "left " << center[0] << " " << center[1]  << "\tright " << center1[0] << " " << center1[1] << endl;
  //cout << "right " << center1[0] << " " << center1[1] << endl;
  imshow("11foo1", publish1);
  cv::namedWindow("22foo");
  imshow("22foo", publish);

  //cv::Rect EyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
  //                        eye_region_top,eye_region_width,eye_region_height);
}

void printMean() {
    cout << endl;
    fout << "meanc = " << rxc / countc << " " << ryc / countc << endl;
    fout << "mean1 = " << rx1 / count1 << " " << ry1 / count1 << endl;
    fout << "meanc = " << rx2 / count2 << " " << ry2 / count2 << endl;
    fout << "meanc = " << rx3 / count3 << " " << ry3 / count3 << endl;
    fout << "meanc = " << rx4 / count4 << " " << ry4 / count4 << endl;
}

void faceDetect(cv::Mat& frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    frame = frame_gray;
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    int i = faces.size()-1;
    if (i == 0)
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        findEyes(faceROI);
        printMean();
    }
    //-- Show what you got
//    imshow( window_name, frame );
}

int main(int argc, char** argv)
{
        //freopen("output_final_hough_eye.txt", "w", stdout);
	// Load the cascade classifiers
	// Make sure you point the XML files to the right path, or
	// just copy the files from [OPENCV_DIR]/data/haarcascades directory
	face_cascade.load("haarcascade_frontalface_alt2.xml");
	eyes_cascade.load("haarcascade_eye.xml");

	// Open webcam
	cv::VideoCapture cap(0);

	// Check if everything is ok
	if (face_cascade.empty() || eyes_cascade.empty() || !cap.isOpened())
		return 1;

	// Set video to 320x240
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	cv::Mat frame, eye_left_tpl, eye_right_tpl, right_eye_frame, left_eye_frame;
	cv::Rect eye_bb1, eye_bb2;

	while (cv::waitKey(15) != 'q')
	{
		cap >> frame;
		if (frame.empty())
			break;

		cv::flip(frame, frame, 1);

		cv::Mat gray;
		//cv::cvtColor(frame, gray, CV_BGR2GRAY);

                faceDetect(frame);

		// Display video
		cv::imshow("video", frame);

		//cv::imshow("right", right_eye_frame);
	}

	return 0;
}

