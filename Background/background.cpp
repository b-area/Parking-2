#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace cv;
using namespace std;

#define MIN_DIST 400
#define MIN_TIME 20

int parking_count = 0;
bool traceParking = false;
bool backProjMode = false;
int vmin = 10, vmax = 256, smin = 30;
int trackObject = 0;

const float A[] = { 1, 1, 0, 1 };
//CvKalman* kalman;
CvMat* state = NULL;
CvMat* measurement;

vector<Point> parking; // polygon that enclose a parking space;
vector<Point> dist;
bool isDist = false;

typedef struct Car
{
    RotatedRect rect;
    Scalar color;
    int time;
    vector<Point> carCenter;
    CvKalman kf;
} Car;

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
    case CV_EVENT_RBUTTONDOWN:
        if (!isDist)
        {
            dist.push_back(Point(x,y));
            cout << "x: " << x << " y: " << y << endl;
            isDist = !isDist;
        }
        else
        {
            Point p1 = dist.back();
            dist.pop_back();
            double distance = sqrtf((x - p1.x)*(x-p1.x) + (y-p1.y)*(y-p1.y));
            cout << "x: " << x << " y: " << y << endl;
            cout << "distance: " << distance << endl;
            isDist = !isDist;
        }
        break;
    case CV_EVENT_RBUTTONUP:
        break;
    }
}

void kalman_filter(CVkalman kalman, float FoE_x, float prev_x)
{
    const CvMat* prediction = cvKalmanPredict( kalman, 0 );
    printf("KALMAN: %f %f %f\n" , prev_x, prediction->data.fl[0] , prediction->data.fl[1] );
    measurement->data.fl[0] = FoE_x;
    cvKalmanCorrect( kalman, measurement);
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
    RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 50, 15 ));
    //trackBox.angle = 90-trackBox.angle;

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
    VideoCapture cap("/home/nick/Desktop/DSC_8474.MOV");
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

        for( unsigned int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect ( Mat(contours_poly[i]) );

            double area = boundRect[i].area();
            //cout << "area: " << area << endl;
            if (area > 65)
            {

                // Track if an object of interest has crossed a roi
                Point2f center(boundRect[i].x + boundRect[i].width/2.0, boundRect[i].y + boundRect[i].height/2.0);

                // Test if the center of a contour has crossed ROI (direction: going in or out)
                if (parking.size() > 3)
                {
                    dist2Center = pointPolygonTest(parking, center, true);
                }
                //cout << center << "is " << dist2Center << " distance from the contour. \n";
                putText(frame, "I", center, FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 1);
                rectangle(frame, boundRect[i], Scalar(255, 0, 0), 2, CV_AA);





                if( trackObject)
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

                    RotatedRect temp = tracking(image, boundRect[i], hist, backproj, hue, mask, phranges);
                    //Draws the temp RotatedRect
                    Point2f vertices[4];
                    temp.points(vertices);
                    for (int k = 0; k < 4; k++)
                    {
                        line(image, vertices[k], vertices[(k+1)%4], Scalar(255,0,0), 1);
                    }

                    bool isClose = false;


                    //Car closestCar = cars[getClosestCar(temp, cars)];


                    for (unsigned int j = 0; j < cars.size(); j++)
                    {
                        double d = getDistance(cars[j].rect, temp);
                        if (d < MIN_DIST)
                        {
                            isClose = true;
                            cars[j].rect = temp;
                            break;
                        }

                    }

                    if (!isClose)
                    {
                        Car candidate;
                        candidate.rect = temp;
                        candidate.color = color;
                        candidate.time = 0;
                        //cout << "adding car" << endl;
                        cars.push_back(candidate);


                        kalman = cvCreateKalman( 2, 1, 0 );
                        state = cvCreateMat( 2, 1, CV_32FC1 );
                        measurement = cvCreateMat( 1, 1, CV_32FC1 );
                        cvSetIdentity( kalman->measurement_matrix,cvRealScalar(1) );
                        memcpy( kalman->transition_matrix->data.fl, A, sizeof(A));
                        cvSetIdentity( kalman->process_noise_cov, cvRealScalar(0.00005) );
                        cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(0.1));
                        cvSetIdentity( kalman->error_cov_post, cvRealScalar(1222));
                        kalman->state_post->data.fl[0] = 0;
                    }
                }

            }
        }

        // Drawing tracked object
        for(unsigned int i = 0; i <cars.size(); i++)
        {
            //if (cars[i].time < MIN_TIME)
                //ellipse(image, cars[i].rect, cars[i].color, 2, CV_AA);

            Point2f vertices[4];
            cars[i].rect.points(vertices);
            for (int k = 0; k < 4; k++)
            {
                line(image, vertices[k], vertices[(k+1)%4], cars[i].color, 2);
            }

            cars[i].time += 1;
            cars[i].carCenter.push_back(cars[i].rect.center);
            if (cars[i].time > MIN_TIME)
            {

                /*if (i == 0)
                {
                    for(unsigned int j = 0; j < cars[i].carCenter.size(); j++)
                    {
                        cout << "cars[" << i << "].carCenter[" << j << "]: " << cars[i].carCenter[j] << endl;
                    }
                    cout << "i: " << i << " time: " << cars[i].time << " current center: " << cars[i].carCenter.back() << " old center: " << cars[i].carCenter[(cars[i].time - MIN_TIME)] << endl;
                    cout << "cars[i].time: " << cars[i].time << " cars[i].time - MIN_TIME: " << (cars[i].time - MIN_TIME) << endl;
                }*/

                //Removes a car from the cars vector if it has not moved in the last MIN_TIME frames.
                if (cars[i].carCenter.back() == cars[i].carCenter[(cars[i].time - MIN_TIME)])
                {
                    cout << "removing car" << endl;
                    cars.erase(cars.begin()+i);
                }
            }
        }
        cout << "cars.size(): " << cars.size() << endl;
        /*
         * Draw parking zone
         */
        for (unsigned int j = 0; j < parking.size(); j++)
        {
            circle(image, parking[j], 5, Scalar(0,0,255), -1);
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
