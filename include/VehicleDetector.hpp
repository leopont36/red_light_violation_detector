/*
 *  VehicleDetector.hpp
 *  Author: Angelica Zonta
 */

#ifndef VEHICLE_DETECTOR_HPP
#define VEHICLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <unordered_map>
#include <string>

using namespace std;
using namespace cv;


class VehicleDetector {

public:
    VehicleDetector(const string& modelPath, float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    std::vector<cv::Rect> detect(const cv::Mat& frame, cv::Rect roi);

private:
    dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
    std::unordered_map<int, std::string> classNames;

    std::vector<cv::Rect> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, int roiYOffset, cv::Size roiSize);
    bool isVehicle(int classId);
    void loadClassNames();
};

#endif