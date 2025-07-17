#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
   
    //day version
    Mat img = imread("../images/vlcsnap-2025-07-15-20h51m19s421.png");

    //night version
    //Mat img = imread("../images/vlcsnap-2025-07-17-00h29m56s656.png");
    
    if (img.empty()) {
        std::cout << "Error: Failed to open the image." << std::endl;
        return -1;
    }

    //cout prints are for debug purposes
    cout << "Image size: " << img.cols << "x" << img.rows << endl;

    //region of interest in video1, video2, video4
    Rect roi(0, 500, 250, 300); // x, y, width, height
    Mat cropped = img(roi);

    //preprocessing for circle detection with Hough
    Mat grayscale, blurred;
    cvtColor(cropped, grayscale, COLOR_BGR2GRAY);
    GaussianBlur(grayscale, blurred, Size(9, 9), 2);

    //detection
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1,
                grayscale.rows / 8, 100, 20, 10, 40);

    // Draw circles back on the original image
    for (int i = 0; i < circles.size(); i++) {
        Vec3f c = circles[i];
        Point center(cvRound(c[0]) + roi.x, cvRound(c[1]) + roi.y);
        int radius = cvRound(c[2]);

        cout<<radius<<endl;

        //circle(img, center, 2, Scalar(0, 0, 255), 3);
        circle(img, center, radius, Scalar(0, 255, 0), 2);

        //take a patch overlapping with the circle
        //TODO: find optimal patch size and a way to detect the most frequent color in a patch
        //this patch size in combination with mean value of the patch doesn't work well

        int patchSize = 5;
        Rect patch(center.x - patchSize, center.y - patchSize, patchSize * 2, patchSize * 2);
        patch.x = max(patch.x, 0);
        patch.y = max(patch.y, 0);
        patch.width = min(patch.width, img.cols - patch.x);
        patch.height = min(patch.height, img.rows - patch.y);

        Mat patchROI = img(patch);

        //converting patch to HSV for color detection
        Mat hsvPatch;
        cvtColor(patchROI, hsvPatch, COLOR_BGR2HSV);

        // Compute average HSV
        Scalar avgHSV = mean(hsvPatch);
        int h = avgHSV[0];
        int s = avgHSV[1];
        int v = avgHSV[2];

        string color = "UNKNOWN";

        cout<<"H: "<<h<<", S: "<<s<<", V: "<<v<<endl;
        if ((h < 15 || h > 160) && s > 100 && v > 100)
            color = "RED";
        //does not work for night images - traffic lights too bright
        else if (h >= 20 && h <= 30 && s > 100 && v > 100)
            color = "YELLOW";
        else if (h >= 40 && h <= 85 && s > 100 && v > 100)
            color = "GREEN";

        cout << "Detected traffic light color: " << color << endl;

        //write the color in the image
        putText(img, color, Point(center.x - 10, center.y - radius - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    }

    imshow("Traffic light detection", img);
    waitKey(0);
    return 0;
   
}






/*
Angelica

#include "VehicleDetector.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // upload the model yolov5 (I had to used an limited versione because it was incompatible with the opncv version of the vlab)
    string modelPath = "../models/yolov5s_clean.onnx"; 
    string videoPath = "../videos/video1.mp4";  
    VehicleDetector detector (modelPath);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) return -1;

    Mat frame;
    while (cap.read(frame)) {

        detector.detect(frame);

        namedWindow("Vehicle Detection", WINDOW_NORMAL);

        imshow("Vehicle Detection", frame);
        if (waitKey(1) == 27) break; 
    }

    return 0;
}





*/
