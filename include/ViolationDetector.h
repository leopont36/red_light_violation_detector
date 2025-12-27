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
    void DetectViolations(VideoCapture cap);
    bool ViolationDetector::DetectViolations(const Mat& img);

private:

    Vec4i FixStopLine(VideoCapture cap);
    TrafficLightDetector traffic_light_detector_;
    VehicleDetector vehicle_detector_;
    StopLineDetector stop_line_detector_;    

};

#endif