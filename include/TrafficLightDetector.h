/*
 *  TrafficLightDetector.h
 *  Author: Milica Masic
 */

#ifndef TRAFFIC_LIGHT_DETECTOR_H
#define TRAFFIC_LIGHT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>

enum class TrafficLightColor {
    Red,
    Yellow,
    Green,
    Unknown
};


class TrafficLightDetector {
public:
    struct DetectionParams {
        double houghParam1 = 100;
        double houghParam2 = 20;
        int minRadius = 10;
        int maxRadius = 40;
        bool useColorThresholdOnly = false;
    };

    TrafficLightDetector(const DetectionParams& params);
    TrafficLightColor DetectTrafficLight(const cv::Mat& img, const cv::Rect& roi);
    void SetParams(const DetectionParams& params);

private:
    DetectionParams params_;
    TrafficLightColor getColorFromPatch(const cv::Mat& patch, const cv::Rect& patchRect, const cv::Mat& img);
    int getMedianHueWithFallback(const cv::Mat& hsv, const cv::Rect& patchRect, const cv::Mat& imgHSV);
};

#endif // TRAFFIC_LIGHT_DETECTOR_H
