#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int parking_count = 0;
bool traceParking = false;
bool backProjMode = false;
int vmin = 10, vmax = 256, smin = 30;
int trackObject = 0;

vector<Point> parking; // polygon that enclose a parking space;

/*
 * Drawing ang getting the rect (polygon) that
 * encircle a parking space.
 */
void onMouse( int event, int x, int y, int flags, void* param )
{
	switch( event )
	{
		case CV_EVENT_LBUTTONDOWN:
			if (traceParking)
			{
				parking.push_back(Point(x,y));
			}
			break;
		case CV_EVENT_LBUTTONUP:
			break;
	} 
}


/*
 * Draw area that is enclosing the parking space.
 *
 */
void drawPaking(Mat &img)
{
	// Draw parking space
	if (!traceParking)
	{
		const Point *pts = (const Point*) Mat(parking).data;
		int npts = Mat(parking).rows;
		polylines(img, &pts,&npts, 1,
				true,           		// draw closed contour (i.e. joint end to start) 
				Scalar(0,255,0),		// colour RGB ordering (here = green) 
				3,              		// line thickness
				CV_AA, 0);
	}
}


int main(int argc, char *argv[])
{
	RNG rng(12345);

	Mat frame;
	Mat back;
	Mat fore;
	Mat image;
	VideoCapture cap(0);
	Rect trackWindow;

	const int nmixtures =3;
	const bool bShadowDetection = false;
	const int history = 4;
	double dist2Center;
	int hsize = 16;
	float hranges[] = {0,180};
	const float* phranges = hranges;
	BackgroundSubtractorMOG2 bg (history,nmixtures,bShadowDetection);

	vector<vector<Point> > contours;
    
	namedWindow( "Histogram", 1 );
	namedWindow("Frame");
	namedWindow("Background");
	setMouseCallback( "Frame", onMouse, 0 );
	createTrackbar( "Vmin", "Frame", &vmin, 256, 0 );
	createTrackbar( "Vmax", "Frame", &vmax, 256, 0 );
	createTrackbar( "Smin", "Frame", &smin, 256, 0 );

	Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

    vector<RotatedRect> cars;
    vector<KalmanFilter> KFList;

	for(;;)
	{
		cap >> frame;
		bg.operator ()(frame,fore);
		bg.getBackgroundImage(back);
		frame.copyTo(image);
		erode(fore,fore,Mat());
		dilate(fore,fore,Mat());

		cvtColor(image, hsv, CV_BGR2HSV);
		findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); // CV_RETR_EXTERNAL retrieves only the extreme outer 

		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );
		vector<Point2f>center( contours.size() );
		vector<float>radius( contours.size() );


        // Update the rotated box tarcking list
        for(unsigned int i = 0; i < cars.size(); i++)
        {
                        
        }


		for( unsigned int i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect ( Mat(contours_poly[i]) );

			double area = boundRect[i].area();
			if (area > 10000)
			{
				
				// Track if an object of interest has crossed a roi
				Point2f center(boundRect[i].x + boundRect[i].width/2.0, boundRect[i].y + boundRect[i].height/2.0);

				// Test if the center of a contour has crossed ROI (direction: going in or out)
				if (parking.size() > 3)
				{
					dist2Center = pointPolygonTest(parking, center, true);
				}
				cout << center << "is " << dist2Center << " distance from the contour. \n"; 
				putText(frame, "I", center, FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 1);
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(100, 100, 200), 2, CV_AA);

				if( trackObject )
				{
					int _vmin = vmin, _vmax = vmax;

					inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
							Scalar(180, 256, MAX(_vmin, _vmax)), mask);
					int ch[] = {0, 0};
					hue.create(hsv.size(), hsv.depth());
					mixChannels(&hsv, 1, &hue, 1, ch, 1);

					Mat roi(hue, boundRect[i]), maskroi(mask, boundRect[i]);
					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					normalize(hist, hist, 0, 255, CV_MINMAX);

					trackWindow = boundRect[i];
					trackObject = 1;

					histimg = Scalar::all(0);
					int binW = histimg.cols / hsize;
					Mat buf(1, hsize, CV_8UC3);
					for( int i = 0; i < hsize; i++ )
						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
					cvtColor(buf, buf, CV_HSV2BGR);

					for( int i = 0; i < hsize; i++ )
					{
						int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
						rectangle( histimg, Point(i*binW,histimg.rows),
								Point((i+1)*binW,histimg.rows - val),
								Scalar(buf.at<Vec3b>(i)), -1, 8 );
					}

					calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
					backproj &= mask;
					RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
					trackBox.angle = 90-trackBox.angle;

					if( backProjMode )
						cvtColor( backproj, image, CV_GRAY2BGR );
                    
				    //ellipse( image, trackBox, color, 3, CV_AA );

                    // drawing a rotated rectangle
                    Point2f rect_points[4]; 
                    trackBox.points( rect_points );
                    for( int j = 0; j < 4; j++ )
                        line( image, rect_points[j], rect_points[(j+1)%4], color, 2, CV_AA );

				}

			}
		}



		/*
		 * Draw parking zone
		 */
		for (unsigned int j = 0; j < parking.size(); j++)
		{
			circle(frame, parking[j], 5, Scalar(0,0,255), -1);
		}
		drawPaking(frame);

		imshow("Frame",image);
		imshow("Background",back);
		imshow( "Histogram", histimg );

		char c = (char)waitKey(30);
		if( c == 27 || c == 'q')
			break;
		switch(c)
		{
			case 's':
				traceParking = !traceParking; 
				break;
			case 't':
				trackObject = !trackObject;
				break;
			case 'b':
				backProjMode = !backProjMode;
				break;
			case 'c':
				parking.clear();
				break;
			default:
				;
		}
	}       

	return 0;
}
