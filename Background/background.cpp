#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "intersection.cpp"

using namespace cv;
using namespace std;

#define MIN_DIST 1000
#define MIN_TIME 2
#define MAX_TIME 10

int parking_count = 0;
bool traceParking = false;
bool backProjMode = false;
int vmin = 10, vmax = 256, smin = 30;
int trackObject = 0;

vector<Point> parking; // polygon that enclose a parking space;

typedef struct Car
{
    RotatedRect rect;
    Scalar color;
    int time;
    KalmanFilter* KF;
    Mat_<float> measurement;

} Car;

/*
* Drawing ang getting the rect (polygon) that
* encircle a parking space.
*/
void onMouse( int event, int x, int y, int flags, void* param )
{
    switch( event )
    {
        case EVENT_LBUTTONDOWN:
            if (traceParking)
            {
                parking.push_back(Point(x,y));
            }
            break;
        case EVENT_LBUTTONUP:
            break;
    }
}

KalmanFilter* CreateKF()
{
KalmanFilter* KF = new KalmanFilter(4, 2, 0);
KF->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);	

    return KF;
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
                true, // draw closed contour (i.e. joint end to start)
                Scalar(0,255,0),	// colour RGB ordering (here = green)
                3, // line thickness
                CV_AA, 0);
    }
}


double getDistance(RotatedRect r1, RotatedRect r2)
{
    Point2f c1 = r1.center;
    Point2f c2 = r2.center;
    Point2f diff = c1 - c2;

    return sqrt(diff.x*diff.x + diff.y*diff.y);
}


int getClosestCar(RotatedRect roi, vector<Car> list)
{
    double min = INFINITY;
    int index = 0;
    for(unsigned int i = 0; i < list.size(); i++)
    {
        double d = getDistance(roi, list[i].rect);
        if (d < min)
        {
            min = d;
            index = i;
        }
    }

    return index;
}


/*
*
*/
RotatedRect tracking(Mat &image, Rect region, Mat& hist, Mat &backproj, Mat& hue, Mat &mask, const float* phranges)
{
    int hsize = 16;

    Mat histimg = Mat::zeros(200, 320, CV_8UC3);

    Rect trackWindow = region;

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
    RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
    trackBox.angle = 90-trackBox.angle;

    if( backProjMode )
        cvtColor( backproj, image, CV_GRAY2BGR );

    //ellipse( image, trackBox, color, 3, CV_AA );

    return trackBox;
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

    vector<Car> cars;
    vector<KalmanFilter> KFList;

    vector<Rect> selections;
    bool isTracking = false;

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

        // Get the rectangle
        for( unsigned int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect ( Mat(contours_poly[i]) );

            double area = boundRect[i].area();
            if (area > 10000)
            {
            	rectangle(image, boundRect[i], Scalar(255,0,0));
            	double width, height;
            	Point2f pts[4];
            	//pts[0] = (boundingRect[i].x, boundingRect[i].y);
            	//pts[1] = (boundingRect[i].x + boundingRect[i].width, boundingRect[i].y);
            	//pts[2] = (boundingRect[i].x, boundingRect[i].y + boundingRect[i].height);
            	//pts[3] = (boundingRect[i].x + boundingRect[i].width, boundingRect[i].y + boundingRect[i].height);
            	Point2f pt1 = boundRect[i].tl();
            	Point2f pt2 = boundRect[i].br();
            	width = pt2.x - pt1.x;
            	height = pt1.y - pt2.y;
            	Point2f pt3 = pt1;
            		    pt3.x = pt1.x + width;
            	Point2f pt4 = pt1;
            			pt1.y = pt1.y - height;
            			
            	pts[0] = pt1;
            	pts[1] = pt2;
            	pts[2] = pt3;
            	pts[3] = pt4;
            	RotatedRect boundRotatedRect;
            	boundRotatedRect.points(pts);
            	
                if (trackObject)
                {
                	bool isClose = false;
                	for (unsigned int j = 0; j < cars.size(); j++)
                	{
                		Point2f vertices[4];
						cars[j].rect.points(vertices);
						for (int k = 0; k < 4; k++)
						{
    						line(image, vertices[k], vertices[(k+1)%4], Scalar(0,0,255));
                		}
                		
                		Mat intersectingRegion;
                		int intersect;
                		intersect = rotatedRectangleIntersection(boundRotatedRect, cars[j].rect, intersectingRegion);
                		
                		
                		if (intersect > 0)
                		{
                			cout << contourArea(intersectingRegion) << "\n";
                			if (contourArea(intersectingRegion) >= 30000)
                			{
                				isClose = true;
                				break;
                			}
                		}
                	}
                    if (!isClose )
                    {
                        cout << "Adding cars to be tracked \n";
                        selections.push_back(boundRect[i]);

                        Car candidate;
                        candidate.color = color;
                        candidate.time = 0;
                        cars.push_back(candidate);
                    }
                }
            }
        }

        if (trackObject)
        {
            int _vmin = vmin, _vmax = vmax;
            inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            mixChannels(&hsv, 1, &hue, 1, ch, 1);

            for (unsigned int j=0; j < selections.size(); j++)
            {
                Mat roi(hue, selections[j]), maskroi(mask, selections[j]);
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                normalize(hist, hist, 0, 255, CV_MINMAX);
                //RotatedRect temp = tracking(image, boundRect[i], hist, backproj, hue, mask, phranges);
                RotatedRect temp = tracking(image, selections[j], hist, backproj, hue, mask, phranges);
                
                
                cars[j].time += 1;
                cars[j].rect = temp;
            }
        }


        // Drawing tracked object
        //cout << cars.size() << " ";
        for(unsigned int j = 0; j <cars.size(); j++)
        {
        	Point2f carVertices[4];
			cars[j].rect.points(carVertices);
			for (int k = 0; k < 4; k++)
    			line(image, carVertices[k], carVertices[(k+1)%4], cars[j].color, 2, 8, 0);
    		ellipse(image, cars[j].rect, cars[j].color, 2);
        	
        }

        /*
		 * Draw parking zone
		 */
        for (unsigned int j = 0; j < parking.size(); j++)
        {
            circle(image, parking[j], 5, Scalar(0,0,255));
        }

        drawPaking(image);

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
                if (traceParking)
                {
                    cout << "In Parking Tracing mode ...\n";
                }
                break;
            case 't':
                trackObject = !trackObject;
                if (trackObject)
                {
                    cout << "In Tracking mode ...\n";
                }
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
