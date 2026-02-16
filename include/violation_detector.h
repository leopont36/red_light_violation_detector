/*
 *  violation_detector.h
 *  Author: Leonardo Pontello
 */

#ifndef VIOLATIONDETECTOR_H
#define VIOLATIONDETECTOR_H

#include <opencv2/opencv.hpp>
#include "trafficlight_detector.h"
#include "vehicle_detector.hpp"
#include "stopline_detector.h"
#include <vector>


class ViolationDetector {
public:
    ViolationDetector(const TrafficlightDetector& traffic_light_detector, const VehicleDetector& vehicle_detector, const StoplineDetector& stop_line_detector);
    void DetectViolationsonVideo(VideoCapture& cap, Rect tl_roi);
    bool DetectViolations(const Mat& img);

private:
    TrafficlightDetector traffic_light_detector_;
    VehicleDetector vehicle_detector_;
    StoplineDetector stop_line_detector_;    
    Rect CalculateROI(const Mat& frame, const Vec4i& stopline); 

};

#endif