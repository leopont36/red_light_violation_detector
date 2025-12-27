/*
 *  ViolationDetector.h
 */

#ifndef VIOLATIONDETECTOR_H
#define VIOLATIONDETECTOR_H

#include <opencv2/opencv.hpp>
#include "TrafficLightDetector.h"
#include "VehicleDetector.hpp"
#include "StopLineDetector.h"
#include <vector>


class ViolationDetector {
public:
    ViolationDetector(const TrafficLightDetector& traffic_light_detector, const VehicleDetector& vehicle_detector, const StopLineDetector& stop_line_detector);
    void DetectViolationsonVideo(VideoCapture& cap, Rect tl_roi);
    bool DetectViolations(const Mat& img);

private:
    TrafficLightDetector traffic_light_detector_;
    VehicleDetector vehicle_detector_;
    StopLineDetector stop_line_detector_;    
    Rect CalculateROI(const Mat& frame, const Vec4i& stopline); 

};

#endif