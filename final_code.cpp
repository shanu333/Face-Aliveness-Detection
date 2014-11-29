#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <cstdio>
#include "constants.h"
#include <queue>
#include <math.h>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <time.h>

#pragma once

#include "cv.h"
#include "highgui.h"
#include "math.h"
#include "list"

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
int f = 0;

Mat eye_left_tpl, eye_right_tpl;
Rect rect1, rect2;
IplImage *ipl;

bool redbox;

CvRect box = cvRect(32, 16, 20, 20);
CvRect box1 = cvRect(32, 16, 20, 20);

int count1, count2, lcount, rcount, ccount, tcount, bcount, realcount;

double leftextremex, leftextremey, rightextremex, rightextremey, plotlx, plotly, plotrx, plotry, plotcx, plotcy;

int poigenx[] = {20, 20, 1260, 1260};
int poigeny[] = {10, 610, 10, 610};

queue <char> que, que2;

vector <int> v;
clock_t strt, ed;
template <typename T>
T abs(T x) {if(x < 0) return -x; return x;}

class voteNode {
public:
	int x;
	int y;
};


void Hough(cv::Mat &imge, double* circle)
{
    equalizeHist( imge, imge );
    IplImage tmp = imge;
    IplImage* img = &tmp;

	uchar* pxl_ptr;
	CvPixelPosition8u pos_src;
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
	int r, x, y, k, cx, cy, max_r = 0, max_radius_vote = 0;
	voteNode temp;
	list<voteNode>::iterator votes;
	bool flag;

	for (i = search_xmin; i < search_xmax; i++)
	{
		for (j = search_ymin; j < search_ymax; j++)
		{
			for (x = x_max/6; x < x_max*5/6; x++)
			{
				for (y = y_max/6; y < y_max*5/6; y++)
				{
					pxl_ptr = CV_MOVE_TO(pos_src, x, y, 1); // Now we point to the ixjth pixel

					if (*pxl_ptr <= 13)	// whites are edges
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
	delete []P;
}

void findEyes(cv::Mat &face, int fileNo) {
    int eye_region_width = face.rows * (kEyePercentWidth/100.0);
    int eye_region_height = face.rows * (kEyePercentHeight/100.0);
    int eye_region_top = face.cols * (kEyePercentTop/100.0);

    cv::Rect LeftEyeRegion(face.rows * (.13), eye_region_top + eye_region_top / 4, face.rows / 2 - face.rows * (.13), eye_region_height - eye_region_top / 2);
   // cv::Rect RightEyeRegion1((face.rows / 2), eye_region_top + eye_region_top / 4, face.rows / 2 - face.rows * (.13), eye_region_height - eye_region_top / 2);
    double Lcircle[3], Rcircle[3];

    cv::Mat Lframe = face(LeftEyeRegion);
    //cv::Mat Rframe = face(RightEyeRegion1);

    // left eye cordinates
    Hough(Lframe, Lcircle);
    circle(Lframe, cv::Point(Lcircle[0], Lcircle[1]), Lcircle[2], 255, 1, 8, 0);
    //cout << Lcircle[0] << " " << Lcircle[1] << endl;
    if (count1 == 2) {

        //if (Lcircle[0] > (rightextremex + leftextremex) / 2)
        if (count2 < 9) {
            if (Lcircle[0] < (leftextremex + rightextremex) / 2) {
                que.push('l');
                lcount++;
            } else if (Lcircle[0] >= (leftextremex + rightextremex) / 2) {
                que.push('r');
                rcount++;
            }

            if (Lcircle[1] <= (leftextremey + rightextremey) / 2) {
                que2.push('t');
                tcount++;
            } else {
                que2.push('b');
                bcount++;
            }

            count2++;
        } else {
            //cout << "name"<<endl;

            if (lcount >= rcount) {
                //box.x = abs((plotlx - leftextremex)) * 640 / (rightextremex - leftextremex);
                box.x = plotlx;

                if (tcount >= bcount) {
                    box.y = plotly;
                } else {
                    box.y = plotry;
                }

            } else if (rcount >= lcount) {
                //box.x = abs((plotrx - leftextremex)) * 640 / (rightextremex - leftextremex);
                box.x = plotrx;
                if (tcount >= bcount) {
                    box.y = plotly;
                } else {
                    box.y = plotry;
                }
            }
            cvRectangle(ipl, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height), cvScalar(0, 255, 0), CV_FILLED);
            cvShowImage("EYE TRACKING", ipl);

            if (que.front() == 'l') {
                lcount--;

            } else if (que.front() == 'r') {
                rcount--;
            }
            if (que2.front() == 't') {
                tcount--;
            } else {
                bcount--;
            }
            que.pop();
            que2.pop();
            if (Lcircle[0] < (leftextremex + rightextremex) / 2) {
                que.push('l');
                lcount++;
            } else if (Lcircle[0] >= (leftextremex + rightextremex) / 2) {
                que.push('r');
                rcount++;
            }

            if (Lcircle[1] <= (leftextremey + rightextremey) / 2) {
                que2.push('t');
                tcount++;
            } else {
                que2.push('b');
                bcount++;
            }
            cvRectangle(ipl, cvPoint(box.x, box.y), cvPoint(box.x+box.width+10,box.y+box.height+10), cvScalar(0, 1, 0), CV_FILLED);
            if (redbox == false) {
                int y = rand() % 4;
                box1.x = poigenx[y];
                box1.y = poigeny[y];
                redbox = true;
            } else {
                if (box.x == box1.x && box1.y == box.y) {
                    cout << "\a";
                    ed = clock();
                    //int nw = ((ed - strt)) / CLOCKS_PER_SEC;
                    cout << ed << " " << strt << " " << ed - strt << endl;

                    if ((int)(ed - strt) >= 0 && (int)(ed - strt) <= 3000) {
                        v[realcount] = 1;
                    } else {
                        v[realcount] = -1;
                    }
                    cout << v[realcount] << endl;
                    int y = rand() % 4;
                    box1.x = poigenx[y];
                    box1.y = poigeny[y];
                    realcount++;
                    strt = ed;

                }
            }
            //cout << que.front() << " " << que2.front() << endl;
            cvRectangle(ipl, cvPoint(box1.x, box1.y), cvPoint(box1.x+box1.width,box1.y+box1.height), cvScalar(0, 0, 255), CV_FILLED);

        }
            //cout << Lcircle[0] - leftextremex << " " << rightextremex - leftextremex << endl;
       // else
        //    box.x = ()


    } else {

        char key = cvWaitKey(10);

        if (char(key) == 32) {
            if (count1 == 0) {
                count1++;
                leftextremex = Lcircle[0];
                leftextremey = Lcircle[1];
                cout << "detected left" << endl;
                cvNamedWindow("done");
                cvDestroyWindow("done");
            } else {
                count1++;
                rightextremex = Lcircle[0];
                rightextremey = Lcircle[1];
                plotlx = 20;
                plotly = 10;
                plotrx = 1260;
                plotry = 610;
                plotcx = 650;
                plotcy = 155;
                cout << "detected right" << endl;
                cvNamedWindow("done");
                cvDestroyWindow("done");
            }
        }
    }

    // right eye cordinates
    //Hough(Rframe, Rcircle);
    //circle(Rframe, cv::Point(Rcircle[0], Rcircle[1]), Rcircle[2], 255, 1, 8, 0);
    //cout << Lcircle[0] << " " << Lcircle[1] << endl;//<< " " << Rcircle[0] << " " << Rcircle[1] << endl;
    cv::namedWindow("LEye");
    imshow("LEye", Lframe);
    //cv::namedWindow("REye");
    //imshow("REye", Rframe);

    //outputToFile(Lcircle, Rcircle, fileNo);
}

void faceDetect(cv::Mat& frame, int fileNumber)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist( frame_gray, frame_gray );
    frame = frame_gray;

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    int i = faces.size()-1;
    if (i == 0)
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        findEyes(faceROI, fileNumber);
    }
}

int main(int argc, char** argv)
{
//    srand(NULL);
    realcount = 0;
    //strt = clock();
    int patterncount = rand() % 5 + 6;
    //freopen("out3.txt", "w", stdout);
    v.resize(patterncount);
    v.clear();
    count1 = count2 = lcount = rcount = ccount = tcount = bcount = 0;
    redbox = false;
    bool authi = false;
	face_cascade.load("haarcascade_frontalface_alt2.xml");
	eyes_cascade.load("haarcascade_eye.xml");
    //gui();
	cv::VideoCapture cap(0);
	if (face_cascade.empty() || eyes_cascade.empty() || !cap.isOpened())
		return 1;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    ipl = cvLoadImage("img.jpg");
    cvNamedWindow("EYE TRACKING", 1);


//	cv::Mat frame, eye_left_tpl, eye_right_tpl, right_eye_frame, left_eye_frame;
//	cv::Rect eye_bb1, eye_bb2;

    Mat frame, frame1;
    double X, Y;
    X = Y = .6;
    int fileNumber;
    cvNamedWindow("Intro");
    while (cv::waitKey(15) != 13) {
        frame1 = imread("img.jpg");
        resize(frame1, frame1, Size(), X, Y, CV_INTER_AREA);
        putText(frame1, "Instructions", cv::Point(550 * X, 50 * Y), CV_FONT_HERSHEY_TRIPLEX, 1.2 * X, cv::Scalar(255, 50, 0, 0),2,8,false);

        putText(frame1, "1. Position the head", cv::Point(50 * X, 100 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.8 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        putText(frame1, "2. Look at the ", cv::Point(50 * X, 130 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "TOP LEFT ", cv::Point(225 * X, 130 * Y), CV_FONT_HERSHEY_TRIPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "corner and press space bar.", cv::Point(360 * X, 130 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        putText(frame1, "3. look at the ", cv::Point(50 * X, 160 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "BOTTOM RIGHT  ", cv::Point(225 * X, 160 * Y), CV_FONT_HERSHEY_TRIPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "corner and press space bar again.", cv::Point(425 * X, 160 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        putText(frame1, "4. Move eyes to ", cv::Point(50 * X, 190 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "RED colored point  ", cv::Point(240 * X, 190 * Y), CV_FONT_HERSHEY_TRIPLEX, 0.7 * X, cv::Scalar(0, 0, 255, 0),1,8,false);
        putText(frame1, "so generated in the screen from the ", cv::Point(475 * X, 190 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);
        putText(frame1, "GREEN colored point.", cv::Point(895 * X, 190 * Y), CV_FONT_HERSHEY_TRIPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        putText(frame1, "5. Continue step 4 till the challenges are completed. ", cv::Point(50 * X, 220 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        putText(frame1, "Press enter to continue......", cv::Point(50 * X, 280 * Y), CV_FONT_HERSHEY_SIMPLEX, 0.7 * X, cv::Scalar(0, 255, 55, 0),1,8,false);

        if (frame1.empty()) break;
        imshow("Intro", frame1);
    }
    cvDestroyWindow("Intro");
	while (cv::waitKey(15) != 'q')
	{
	    if (patterncount == realcount) {
	        int c1 = 0, c0 = 0;
	        for (int i = 1; i < patterncount; i++) {
                if(v[i] == 1) c1++;
                else c0++;
                cout << v[i] << " ";

	        }
	        if ((double)c1 / (double)(patterncount - 1) > 0.300000) {
                    cout << "authenticated\n";
                    authi = true;
            } else {
                    cout << "not authenticated\n";
                    authi = false;
            }
            //cout << c1 << " " << patterncount - 1 << " " << (double)c1 / (double)(patterncount - 1) << endl;
            break;
	    }
	    /*clock_t start = clock();
	    srand(time(NULL));
	    fileNumber = rand() % 4;
	    cout << "fileNumber = " << fileNumber << endl;
	    while ((float)clock() - (float)start <= 2 * CLOCKS_PER_SEC) {}
        */cap >> frame;
        if (frame.empty())
            break;

        cv::flip(frame, frame, 1);

        faceDetect(frame, fileNumber);


		cv::imshow("video", frame);
    }
    cvNamedWindow("Intro");
    char *sss;

    if (authi) {
        sss = "The user has been successfully verfied to be live......";
    } else {
        sss = "The user has been verfied to be not live......";
    }
    X = Y = .6;
    while (cv::waitKey(15) != 13) {
        frame1 = imread("img.jpg");
        resize(frame1, frame1, Size(), X, Y, CV_INTER_AREA);
        putText(frame1, sss, cv::Point(50 * X, 100 * Y), CV_FONT_HERSHEY_TRIPLEX, 1.2 * X, cv::Scalar(0, 255, 55, 0),2,8,false);
        if (frame1.empty()) break;
        imshow("Intro", frame1);
    }
    cout << leftextremex << " " << leftextremey << " " << rightextremex << " " << rightextremey << endl;

	return 0;
}
