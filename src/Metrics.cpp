/*
 *  metrics.cpp
 *  Author: Leonardo Pontello
 */

#include "metrics.h"

using namespace std;
using namespace cv;

Metrics::Metrics(const TrafficLightDetector& traffic_light_detector, const VehicleDetector& vehicle_detector, const StopLineDetector& stop_line_detector, const ViolationDetector& violation_detector)
    : traffic_light_detector_(traffic_light_detector), vehicle_detector_(vehicle_detector), stop_line_detector_(stop_line_detector), violation_detector_(violation_detector)
{}

void Metrics::ComputeMetricsFrame(Mat frame, GroundTruthData data)
{
    Mat visualization = frame.clone();
    
    TrafficLightColor color = traffic_light_detector_.DetectTrafficLight(frame, Rect(0, 0, frame.cols, frame.rows));
    if(data.light_color == color)
    {
        cout << "Traffic light: CORRECT" << endl;
        correct_++;
    }
    else
        cout << "Traffic light: WRONG" << endl;
    classifications_++;

    if(data.has_stopline)
    {
        Rect stopline = stop_line_detector_.detectStopLine(frame);
        double IoU = ComputeIoU(stopline, data.stopline);
        
        stopline_iou_sum_ += IoU;  
        stopline_count_++;          
        
        cout << "Stopline IoU: " << IoU << endl;
        
        rectangle(visualization, stopline, Scalar(0, 255, 0), 2);
    }

    if(data.vehicles.size() > 0)
    {
        vector<Rect> vehicles = vehicle_detector_.detect(frame, Rect(0, 0, frame.cols, frame.rows));
        const double IOU_THRESHOLD = 0.5;
        
        int truePositives = 0;
        vector<bool> gtMatched(data.vehicles.size(), false);
        
        for(const auto& detectedVehicle : vehicles)
        {
            double maxIoU = 0.0;
            int maxIdx = -1;
            
            for(size_t i = 0; i < data.vehicles.size(); i++)
            {
                if(gtMatched[i]) continue;
                double iou = ComputeIoU(detectedVehicle, data.vehicles[i]);
                if(iou > maxIoU)
                {
                    maxIoU = iou;
                    maxIdx = i;
                }
            }
            
            if(maxIoU >= IOU_THRESHOLD && maxIdx >= 0)
            {
                truePositives++;
                gtMatched[maxIdx] = true;
            }
            
            rectangle(visualization, detectedVehicle, Scalar(255, 0, 0), 2);
        }
        
        double AP = (vehicles.size() > 0) ? static_cast<double>(truePositives) / vehicles.size() : 0.0;
        
        vehicle_ap_sum_ += AP;  
        vehicle_frames_++;       
        
        cout << "Vehicles detected: " << vehicles.size() << endl;
        cout << "Vehicles AP: " << AP << endl;
    }

    bool violation = violation_detector_.DetectViolations(frame);

    if(violation && data.has_violation)
    {
        cout << "Violation: TP" << endl;
        v_tp++;
    }
    else if(violation && !data.has_violation)
    {
        cout << "Violation: FP" << endl;
        v_fp++;
    }
    else if(!violation && data.has_violation)
    {
        cout << "Violation: FN" << endl;
        v_fn++;
    }
    else
    {
        cout << "Violation: TN" << endl;
        v_tn++;
    }
    
    imshow("Detection Results", visualization);
    waitKey(0);
}

void Metrics::ComputeMetrics()
{
    if(classifications_ > 0) {
        double tl_acc = static_cast<double>(correct_) / classifications_;
        cout << "Traffic Light Accuracy: " << tl_acc << endl;
    }

    if(stopline_count_ > 0) {
        double sl_mIoU = stopline_iou_sum_ / stopline_count_;
        cout << "Stopline mean IoU: " << sl_mIoU << endl;
    }

    if(vehicle_frames_ > 0) {
        double vehicle_mAP = vehicle_ap_sum_ / vehicle_frames_;
        cout << "Vehicle mAP: " << vehicle_mAP << endl;
    }

    int total_violations = v_tp + v_tn + v_fp + v_fn;
    if(total_violations > 0) {
        double violation_acc = static_cast<double>(v_tp + v_tn) / total_violations;
        cout << "Violation Detection Accuracy: " << violation_acc << endl;
    }
}

double Metrics::ComputeIoU(const Rect& rect1, const Rect& rect2)
{
    Rect intersection = rect1 & rect2;
    
    if(intersection.area() == 0)
        return 0.0;
    
    double intersectionArea = intersection.area();
    double unionArea = rect1.area() + rect2.area() - intersectionArea;
    
    return intersectionArea / unionArea;
}

string Metrics::ExtractString(const string& json, size_t start) {
    size_t first_quote = json.find("\"", start + 8); // after "label":
    size_t second_quote = json.find("\"", first_quote + 1);
    return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

double Metrics::ExtractNumber(const string& json, const string& key, size_t start) {
    size_t pos = json.find(key, start);
    if (pos == string::npos) return 0;
    
    pos += key.length();
    while (json[pos] == ' ') pos++;
    
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-')) {
        end++;
    }

    return stod(json.substr(pos, end - pos));
}

Rect Metrics::ExtractRect(const string& json, size_t start) {
    size_t rect_pos = json.find("\"rect\":", start);
    
    double x = ExtractNumber(json, "\"x\":", rect_pos);
    double y = ExtractNumber(json, "\"y\":", rect_pos);
    double w = ExtractNumber(json, "\"width\":", rect_pos);
    double h = ExtractNumber(json, "\"height\":", rect_pos);
    
    return Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h));
}

Metrics::GroundTruthData Metrics::Parse(const string& json) {
    GroundTruthData data;    
    size_t pos = 0;
    
    while ((pos = json.find("\"label\":", pos)) != string::npos) {
        string label = ExtractString(json, pos);
            
        if (label == "vehicle") {
            Rect rect = ExtractRect(json, pos);
            data.vehicles.push_back(rect);
                
            // Check if violation: true
            size_t viol_pos = json.find("\"violation\":", pos);
            size_t next_label = json.find("\"label\":", pos + 1);
            if (viol_pos != string::npos && (next_label == string::npos || viol_pos < next_label)) {
                if (json.find("true", viol_pos) != string::npos && json.find("true", viol_pos) < json.find("}", viol_pos)) 
                {
                    data.has_violation = true;
                }
            }
        }
        else if (label == "stop_line") {
            data.stopline = ExtractRect(json, pos);
            data.has_stopline = true;
        }
        else if (label == "red_light") {
            data.light_color = TrafficLightColor::Red;
        }
        else if (label == "yellow_light") {
            data.light_color = TrafficLightColor::Yellow;
        }
        else if (label == "green_light") {
            data.light_color = TrafficLightColor::Green;
        }
           
        pos++;
    }
        
    return data;
}