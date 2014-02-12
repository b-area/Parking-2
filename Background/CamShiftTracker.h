#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

class CamShiftTracker 
{
    private:
        Mat hist_img = Mat::zeros(200, 320, CV_8UC3);
        Mat back_proj;

        int hsize;
        const float* phranges;;

        Rect trackWindow;
        RotatedRect trackBox;

    public:
        Mat getBackProjection(){return back_proj;}
        Rect getTrackWindow(){return trackWindow;}
        RotatedRect getTrackBox(){return trackBox;}

        /* Constructors */
        CamShiftTracker();
        CamShiftTracker(Mat frame, Rect areaToTrack);

        /* Default constructor */
        ~CamShiftTracker();

        /* Method to track a boxed blob (trackWindow) within a frame */
        void track(Mat frame, Rect selection, int vmin, int vmax, int smin); 
};

/* Default Constructors */
CamShiftTracker::CamShiftTracker()
{
    hsize = 16;
    float hranges[] = {0,180};
    phranges = hranges;
    hist_img = Mat::zeros(200, 320, CV_8UC3);
}

CamShiftTracker::~CamShiftTracker()
{

}

/* Tracking a window (boxed blob) */
void CamShiftTracker::track(Mat frame, Rect selection, int vmin, int vmax, int smin)
{
    Mat image, hsv, hue, mask, hist;

    frame.copyTo(image);
    cvtColor(image, hsv, CV_BGR2HSV);

    int _vmin = vmin, _vmax = vmax;
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    Mat roi(hue, selection), maskroi(mask, selection);
    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

    trackWindow = selection; /*WATCH CLOSELY*/

    hist_img = Scalar::all(0);
    int binW = hist_img.cols / hsize;
    Mat buf(1, hsize, CV_8UC3);
    for( int i = 0; i < hsize; i++ )
        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
    cvtColor(buf, buf, CV_HSV2BGR);

    for( int i = 0; i < hsize; i++ )
    {
        int val = saturate_cast<int>(hist.at<float>(i)*hist_img.rows/255);
        rectangle( hist_img, Point(i*binW,hist_img.rows),
                Point((i+1)*binW,hist_img.rows - val),
                Scalar(buf.at<Vec3b>(i)), -1, 8 );
    }

    calcBackProject(&hue, 1, 0, hist, back_proj, &phranges);
    back_proj &= mask;
    RotatedRect trackBox = CamShift(back_proj, trackWindow,
            TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
    trackBox.angle = 90-trackBox.angle;

}
