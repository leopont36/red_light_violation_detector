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

void ViolationDetector::DetectViolationsonVideo(VideoCapture& cap, Rect tl_roi)
{
    Mat frame;
    cap.read(frame);
    Vec4i stopline = stop_line_detector_.detectStopLine(frame);
    
    int y_line = (stopline[1] + stopline[3]) / 2;
    Rect roi = CalculateROI(frame, stopline);

    namedWindow("Detection", WINDOW_NORMAL);

    while(cap.read(frame))
    {
        Mat display = frame.clone();
        
        // Disegna stopline
        line(display, Point(stopline[0], stopline[1]), Point(stopline[2], stopline[3]), Scalar(0, 255, 0), 3);
        
        // Disegna ROI
        rectangle(display, roi, Scalar(255, 255, 0), 2);
        
        TrafficLightColor light = traffic_light_detector_.DetectTrafficLight(frame, tl_roi);
        string light_text = (light == TrafficLightColor::Red) ? "RED" : "NOT RED";
        putText(display, light_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        vector<Rect> vehicles = vehicle_detector_.detect(frame, roi);
        
        putText(display, "Vehicles: " + to_string(vehicles.size()), Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
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
}

bool ViolationDetector::DetectViolations(const Mat& img)
{
    Vec4i stopline = stop_line_detector_.detectStopLine(img);
    TrafficLightColor color = traffic_light_detector_.DetectTrafficLight(img, Rect(0, 0, img.cols, img.rows));
    
    // Calcola la y media della stopline
    int stopline_y = (stopline[1] + stopline[3]) / 2;
    
    // Usa la stessa funzione per calcolare la ROI
    
    vector<Rect> vehicles = vehicle_detector_.detect(img, CalculateROI(img, stopline));

    // Nessuna violazione se non c'è semaforo rosso
    if(color != TrafficLightColor::Red)
        return false;

    // Nessuna violazione se non ci sono veicoli o stopline non valida
    if(stopline[0] == 0 && stopline[1] == 0 && stopline[2] == 0 && stopline[3] == 0)
        return false;
    
    if(vehicles.empty())
        return false;
    
    // Controlla se qualche veicolo ha superato la stopline
    for(const auto& vehicle : vehicles)
    {
        int vehicle_center_y = vehicle.y + vehicle.height / 2;
        if(vehicle_center_y < stopline_y)  // Vehicle crossed the line
            return true;
    }
    
    return false;
}

Rect ViolationDetector::CalculateROI(const Mat& frame, const Vec4i& stopline) 
{
    int y_line = (stopline[1] + stopline[3]) / 2;
    int roi_height = static_cast<int>(frame.rows * 0.4);
    
    // ROI centered on the stopline
    int roi_y = y_line - roi_height / 2;
    roi_y = max(0, roi_y);
    
    int roi_x = min(stopline[0], stopline[2]);
    int roi_width = abs(stopline[0] - stopline[2]);
    
    Rect roi = Rect(roi_x, roi_y, roi_width, roi_height);
    
    // Clip ROI to frame boundaries
    roi = roi & Rect(0, 0, frame.cols, frame.rows);
    
    return roi;
}
