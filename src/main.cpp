#include "StopLineDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    //string filename = "C:\\Users\\Principale\\Desktop\\image1.png";
    //string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m19s421.png";
    string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-17-00h29m56s656.png";
    // Load image
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    // Detect stop line
    StopLineDetector detector;
    Vec4i stopLine = detector.detectStopLine(img);

    // Draw stop line
    line(img, Point(stopLine[0], stopLine[1]), Point(stopLine[2], stopLine[3]), Scalar(0, 255, 0), 3);

    // Show result
    namedWindow("Stop Line Detection", WINDOW_NORMAL);
    imshow("Stop Line Detection", img);
    waitKey(0);

    return 0;
}