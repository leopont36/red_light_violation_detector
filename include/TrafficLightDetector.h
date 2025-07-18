//Author: Milica Masic

#ifndef TRAFFIC_LIGHT_DETECTOR_H
#define TRAFFIC_LIGHT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>

class TrafficLightDetector {
public:
    // Struct defined inside the class
    struct DetectionParams {
        double houghParam1 = 100;
        double houghParam2 = 20;
        int minRadius = 10;
        int maxRadius = 40;
        bool useColorThresholdOnly = false;
        cv::Rect roi = cv::Rect(); // region of interest
    };

    // Constructor that accepts parameters
    TrafficLightDetector(const DetectionParams& params);

    void detectAndAnnotate(cv::Mat& img, const std::string& videoName);

private:
    DetectionParams params_;
    //const methods?
    cv::Rect getSafeROI(const cv::Mat& img, const std::string& videoName);
    std::string getColorFromPatch(const cv::Mat& patch, const cv::Rect& patchRect, const cv::Mat& img);
    int getMedianHueWithFallback(const cv::Mat& hsv, const cv::Rect& patchRect, const cv::Mat& imgHSV);
};

#endif // TRAFFIC_LIGHT_DETECTOR_H
