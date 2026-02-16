/*
 *  ViolationDetector.cpp
 *  Author: Leonardo Pontello
 */

#include "violation_detector.h"
#include <cmath>

using namespace cv;
using namespace std;

ViolationDetector::ViolationDetector(const TrafficlightDetector& traffic_light_detector,
    const VehicleDetector& vehicle_detector,
    const StoplineDetector& stop_line_detector) 
    : traffic_light_detector_(traffic_light_detector),
      vehicle_detector_(vehicle_detector),
      stop_line_detector_(stop_line_detector)
{}

static bool isStopLineValid(const Vec4i& stopline) {
    return !(stopline[0] == 0 && stopline[1] == 0 && stopline[2] == 0 && stopline[3] == 0);
}

void ViolationDetector::DetectViolationsonVideo(VideoCapture& cap, Rect tl_roi)
{
    Mat frame;

    // read first frame to detect stop line position
    if (!cap.read(frame))
        return;

    Vec4i stopline = stop_line_detector_.detectStopline(frame);
    if (!isStopLineValid(stopline))
        return;

    int y_line = (stopline[1] + stopline[3]) / 2;
    Rect roi = CalculateROI(frame, stopline);

    // stopline width could be 0 if detection failed
    if (roi.empty())
        return;

    namedWindow("Detection", WINDOW_NORMAL);

    while (cap.read(frame))
    {
        Mat display = frame.clone();

        line(display, Point(stopline[0], stopline[1]), Point(stopline[2], stopline[3]), Scalar(0, 255, 0), 3);
        rectangle(display, roi, Scalar(255, 255, 0), 2);

        TrafficlightColor light = traffic_light_detector_.DetectTrafficlight(frame, tl_roi);
        string light_text = (light == TrafficlightColor::Red) ? "RED" : "NOT RED";
        putText(display, light_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        vector<Rect> vehicles = vehicle_detector_.detect(frame, roi);
        putText(display, "Vehicles: " + to_string(vehicles.size()), Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        for (const auto& vehicle : vehicles)
        {
            int vehicle_center_y = vehicle.y + vehicle.height / 2;
            Scalar color = (vehicle_center_y < y_line) ? Scalar(0, 0, 255) : Scalar(255, 0, 0);

            rectangle(display, vehicle, color, 2);
            circle(display, Point(vehicle.x + vehicle.width/2, vehicle_center_y), 5, color, -1);

            if (vehicle_center_y < y_line && light == TrafficlightColor::Red)
                putText(display, "VIOLATION", Point(vehicle.x, vehicle.y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        imshow("Detection", display);
        waitKey(1);
    }
}

bool ViolationDetector::DetectViolations(const Mat& img)
{
    Vec4i stopline = stop_line_detector_.detectStopline(img);

    // check preconditions before running heavy detectors
    if (!isStopLineValid(stopline))
        return false;

    TrafficlightColor color = traffic_light_detector_.DetectTrafficlight(img, Rect(0, 0, img.cols, img.rows));
    if (color != TrafficlightColor::Red)
        return false;

    Rect roi = CalculateROI(img, stopline);
    if (roi.empty())
        return false;

    vector<Rect> vehicles = vehicle_detector_.detect(img, roi);
    if (vehicles.empty())
        return false;

    int stopline_y = (stopline[1] + stopline[3]) / 2;

    for (const auto& vehicle : vehicles)
    {
        int vehicle_center_y = vehicle.y + vehicle.height / 2;
        if (vehicle_center_y < stopline_y)
            return true;
    }

    return false;
}

Rect ViolationDetector::CalculateROI(const Mat& frame, const Vec4i& stopline)
{
    int y_line = (stopline[1] + stopline[3]) / 2;
    int roi_height = static_cast<int>(frame.rows * 0.4);

    int roi_y = max(0, y_line - roi_height / 2);
    int roi_x = min(stopline[0], stopline[2]);
    int roi_width = abs(stopline[0] - stopline[2]);

    Rect roi(roi_x, roi_y, roi_width, roi_height);
    roi &= Rect(0, 0, frame.cols, frame.rows);

    return roi;
}