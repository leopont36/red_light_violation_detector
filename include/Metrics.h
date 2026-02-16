/*
 *  metrics.h
 *  Author: Leonardo Pontello
 */

#ifndef METRICS_H
#define METRICS_H

#include <opencv2/opencv.hpp>
#include "trafficlight_detector.h"
#include "vehicle_detector.hpp"
#include "stopline_detector.h"
#include "violation_detector.h"

#include <vector>
#include <string>

using namespace std;
using namespace cv;

class Metrics {
public:
    struct GroundTruthData {
        vector<Rect> vehicles;
        bool has_violation;
        Rect stopline;
        bool has_stopline;
        TrafficlightColor light_color;
        GroundTruthData() : has_violation(false), has_stopline(false), light_color(TrafficlightColor::Unknown) {}
    };

    Metrics(const TrafficlightDetector& traffic_light_detector, const VehicleDetector& vehicle_detector, const StoplineDetector& stop_line_detector, const ViolationDetector& violation_detector);
    void ComputeMetricsFrame(Mat Frame, GroundTruthData data);
    void ComputeMetrics();
    static GroundTruthData Parse(const string& json);

private:
    static double ComputeIoU(const Rect& rect1, const Rect& rect2);
    static string ExtractString(const string& json, size_t start);
    static double ExtractNumber(const string& json, const string& key, size_t start);
    static Rect ExtractRect(const string& json, size_t start);


    TrafficlightDetector traffic_light_detector_;
    VehicleDetector vehicle_detector_;
    StoplineDetector stop_line_detector_;
    ViolationDetector violation_detector_;

    // Traffic Light Detection
    int correct_ = 0;
    int classifications_ = 0;

    // StopLine Detection
    double stopline_iou_sum_ = 0.0;
    int stopline_count_ = 0;
    
    // Vehicle Detection
    double vehicle_ap_sum_ = 0.0;
    int vehicle_frames_ = 0;

    // Violation Detection
    int v_tp = 0;
    int v_tn = 0;
    int v_fn = 0;
    int v_fp = 0;
};

#endif // METRICS_H