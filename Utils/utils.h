#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

class CamShitTracker 
{
    private:
        Mat hsv;
        Mat hue;
        Mat mask;
        Mat hist;
        Mat histimg; //= Mat::zeros(200, 320, CV_8UC3)
        Mat back_proj;
    
    public:
};

/*
 * Notice: Use the double cv::pointPolygonTest(const Mat& contour, Point2f pt, bool measureDist) 
 * isntead
 *
 * Point inclusion in Polygon Test by W. Randolph Franklin.
 * Shamelessly copied and pasted from:
 * http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#ack
 *
 * Parameters:
 *  + nvert: number of vertices in the polygon.
 *  + vertx, verty: arrays that have the lists of (x,y) coordinates.  
 *  + testx, testy: the (x,y) coordinate to be tested.
 */
int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
{
    int i, j, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
                (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
            c = !c;
    }
    return c;
}
