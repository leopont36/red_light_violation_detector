/*
 *  VehicleDetector.hpp
 *  Author: Angelica Zonta
 */

#ifndef VEHICLE_DETECTOR_HPP
#define VEHICLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;


class VehicleDetector {

public:
    VehicleDetector(const string& modelPath, float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    void detect(Mat& frame);

private:
    dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
    vector<string> classNames;

    //void postprocess(Mat& frame, const vector<Mat>& outs);
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, int roiYOffset, cv::Size roiSize);
    bool isVehicle(int classId);
    void loadClassNames(); 
};

#endif
