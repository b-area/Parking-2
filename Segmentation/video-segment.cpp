#include "opencv2/opencv.hpp"
#include <string>

using namespace cv;
//using namespace std;

/* Parameters */
int thresh = 100;
int max_thresh = 255;
int n_erode = 0;
int n_dilate = 0;
int max_erode = 20;
int max_dilate = 20;

cv::Mat image;
cv::Mat src_gray;
cv::Mat orig_frame;
cv::Mat curr_frame;
cv::Mat prev_frame;
cv::Mat diff_frame;

cv::RNG rng(12345);    

/* 
 * Watershed Segmenter class
 * to be used to segment images
 */
class WatershedSegmenter
{
private:
    cv::Mat markers;
public:
    void setMarkers(cv::Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};

/* Function Headers */
void thresh_callback(int, void* );
void findDifference(cv::Mat, cv::Mat, cv::Mat &, int);

/* Main */
int main(int argc, char* argv[])
{
    // Get image from camera
    cv::VideoCapture cap(0);

    // Get the first frame
    orig_frame = cv::imread(argv[1]);

    /*
     * Somehow cap >> orig_frame 
     * keeps changing the orig_frame
     * in the infinite loop below (I guess >> returns a pointer to the most recent frame)
     * i.e. it changes every time a new frame comes in.
     */

    //cv::waitKey(2000);
    // Create the first frame
    //cap >> orig_frame;
    //cv::imwrite("orig-frame.jpg", orig_frame);

    for(;;)
    {
        cap >> curr_frame;

        //diff_frame = cv::abs(curr_frame - orig_frame);
        findDifference(curr_frame, orig_frame, diff_frame, 100);
        diff_frame.copyTo(image);
        cv::cvtColor(image, src_gray, CV_BGR2GRAY);

        /// Create Window
        const char* source_window = "Source";
        namedWindow( source_window, WINDOW_AUTOSIZE );
        imshow( source_window, curr_frame);

        /* Parameters setters */ 
        createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
        createTrackbar( " Erode:", "Source", &n_erode, max_erode, thresh_callback );
        createTrackbar( " Dilate:", "Source", &n_dilate, max_dilate, thresh_callback );
        thresh_callback( 0, 0 );
        waitKey(200);
    }
    
    return 0;
}


void thresh_callback(int, void* )
{
    cv::Mat binary;

    cv::threshold(src_gray, binary, thresh, 255, THRESH_BINARY);

    // Eliminate noise and smaller objects
    cv::Mat fg;
    cv::erode(binary,fg,cv::Mat(),cv::Point(-1,-1),n_erode);
    //imshow("fg", fg);

    // Identify image pixels without objects
    cv::Mat bg;
    cv::dilate(binary,bg,cv::Mat(),cv::Point(-1,-1), n_dilate);
    cv::threshold(bg,bg,1, 128,cv::THRESH_BINARY_INV);
    //imshow("bg", bg);

    // Create markers image
    cv::Mat markers(binary.size(),CV_8U,cv::Scalar(0));
    markers= fg+bg;
    //imshow("markers", markers);

    // Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);

    cv::Mat result = segmenter.process(image);
    result.convertTo(result,CV_8U);

    // --------------------------
    // Analyze blobs -- contours
    // --------------------------
    //Detect edges using canny
    cv::Mat canny_output;
    std::vector<Vec4i> hierarchy;
    cv::Canny(result, canny_output, thresh, thresh*2, 3 );
    
    // Find contours
    std::vector<std::vector<cv::Point> > contours;  
    cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //cv::findContours(canny_output,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); // CV_RETR_EXTERNAL retrieves only the extreme outer contours
    //cv::drawContours(result,contours,-1,cv::Scalar(0,0,255),2); // This draws all the contours
   
    // Only draw the important contours
    curr_frame.copyTo(prev_frame);
    for( int i = 0; i< contours.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        double area = contourArea(contours[i]);
        if (area > 15.0)
        {
            //printf("Area[%d]: %.2f\n", i, contourArea(contours[i]));
            cv::drawContours(prev_frame,contours,i,color,2); // This draws all the contours
        }
    }   
   
    imshow("final_result", result);
    imshow("Contours", prev_frame);
}

/* Get the difference between two images */
void findDifference(cv::Mat src1, cv::Mat src2, cv::Mat &dst, int threshold) {
        dst = cv::abs(src2 - src1);
        cv::threshold(dst, dst, threshold, 255, cv::THRESH_BINARY);
}
