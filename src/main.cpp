/*
 *  main.cpp
 *  Author: Milica Masic
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include "TrafficLightDetector.h"
#include "VehicleDetector.hpp"

using namespace cv;
using namespace std;

// select detection parameters based on video
TrafficLightDetector::DetectionParams getDetectionParamsForVideo(
    const std::string& videoName, int cols, int rows) {
    
    TrafficLightDetector::DetectionParams params;

    if (videoName == "video1.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video2.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video3.mp4") {
        params.houghParam1 = 120;
        params.houghParam2 = 25;
        params.minRadius = 8;
        params.maxRadius = 35;
        params.roi = Rect(1670, 170, 90, 210);
    } else if (videoName == "video4.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video5.mp4") {
        params.houghParam1 = 30;
        params.houghParam2 = 10;
        params.minRadius = 2;
        params.maxRadius = 10;
        params.roi = Rect(290, 70, 15, 25);
    } else {
        //default
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(0, 0, cols, rows);
    }

    return params;
}

int main(int argc, char** argv)
{
    //video display
    string videoPath = "../videos/video1.mp4";  
    string videoName = videoPath.substr(videoPath.find_last_of("/\\") + 1);

    //upload the model yolov5 (we had to used an limited versione because it was incompatible with the opncv version of the vlab)
    string modelPath = "../models/yolov5s_clean.onnx"; 
    VehicleDetector detector (modelPath, 0.3f, 0.45f);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) return -1;

    Mat frame;
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = 1000 / fps;  //delay between frames in ms

    namedWindow("Traffic light detection", WINDOW_NORMAL);

    TrafficLightDetector::DetectionParams params = getDetectionParamsForVideo(videoName, frame.cols, frame.rows);
    TrafficLightDetector tlDetector(params);

    while (cap.read(frame)) {

        detector.detect(frame);
        tlDetector.detectAndAnnotate(frame, videoName);
        
        imshow("Traffic light detection", frame);
        if (waitKey(delay) == 27) break;
    }
    destroyAllWindows();

    return 0;
   
}