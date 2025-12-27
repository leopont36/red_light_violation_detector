/*
 *  ViolationDetector.cpp
 *  Author: Leonardo Pontello
 */

#include "ViolationDetector.h"
#include <cmath>

using namespace cv;
using namespace std;

ViolationDetector::ViolationDetector(const TrafficLightDetector& traffic_light_detector, const VehicleDetector& vehicle_detector, const StopLineDetector& stop_line_detector) : 
    traffic_light_detector_(traffic_light_detector), vehicle_detector_(vehicle_detector), stop_line_detector_(stop_line_detector)
{}
/*
void ViolationDetector::DetectViolations(VideoCapture cap)
{
    Rect stopline = FixStopLine(cap);
    cap.set(CAP_PROP_POS_FRAMES, 0);

    Mat frame;
    cap.read(frame);
    
    int y_line = (stopline[1] + stopline[3]) / 2;
    int roi_height = static_cast<int>(frame.rows * 0.4);
    
    // ROI centrata sulla stopline
    int roi_y = y_line - roi_height / 2;
    roi_y = max(0, roi_y);  // Non andare sotto 0
    
    int roi_x = min(stopline[0], stopline[2]);
    int roi_width = abs(stopline[0] - stopline[2]);
    
    Rect roi = Rect(roi_x, roi_y, roi_width, roi_height);
    roi = roi & Rect(0, 0, frame.cols, frame.rows);
    
    int frame_num = 0;
    
    namedWindow("Detection", WINDOW_NORMAL);

    while(cap.read(frame))
    {
        frame_num++;
        Mat display = frame.clone();
        
        // Disegna stopline
        line(display, Point(stopline[0], stopline[1]), Point(stopline[2], stopline[3]), Scalar(0, 255, 0), 3);
        
        // Disegna ROI
        rectangle(display, roi, Scalar(255, 255, 0), 2);
        
        TrafficLightColor light = traffic_light_detector_.DetectTrafficLight(frame);
        string light_text = (light == TrafficLightColor::Red) ? "RED" : "NOT RED";
        putText(display, light_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        vector<Rect> vehicles = vehicle_detector_.detect(frame, roi);
        
        putText(display, "Vehicles: " + to_string(vehicles.size()), Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(display, "Frame: " + to_string(frame_num), Point(10, 110), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        for (const auto& vehicle : vehicles)
        {
            int vehicle_center_y = vehicle.y + vehicle.height / 2;
            Scalar color = (vehicle_center_y < y_line) ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
            
            rectangle(display, vehicle, color, 2);
            circle(display, Point(vehicle.x + vehicle.width/2, vehicle_center_y), 5, color, -1);
            
            if (vehicle_center_y < y_line && light == TrafficLightColor::Red)
            {
                putText(display, "VIOLATION", Point(vehicle.x, vehicle.y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
        }
        
        imshow("Detection", display);
        waitKey(1);
    }
}*/

bool ViolationDetector::DetectViolations(const Mat& img)
{
    Rect stopline = stop_line_detector_.detectStopLine(img);
    TrafficLightColor color = traffic_light_detector_.DetectTrafficLight(img, Rect(0, 0, img.cols, img.rows));
    int roiX = max(0, stopline.x);
    int roiY = max(0, stopline.y);
    int roiWidth = min(img.cols - roiX, img.cols - stopline.x);
    int roiHeight = min(img.rows - roiY, static_cast<int>(img.rows * 0.4));

    Rect validROI(roiX, roiY, roiWidth, roiHeight);
    vector<Rect> vehicles = vehicle_detector_.detect(img, validROI);

    if(color != TrafficLightColor::Red)
        return false;

    if(stopline.area() == 0 || vehicles.empty())
        return false;
    
    int stopline_y = stopline.y + stopline.height;
    
    for(const auto& vehicle : vehicles)
    {
        int vehicle_bottom = vehicle.y + vehicle.height;
        
        if(vehicle_bottom > stopline_y)
            return true;
    }
    
    return false;

}
