#include <opencv2/opencv.hpp>
#include "violation_detector.h"
#include "vehicle_detector.hpp"
#include "trafficlight_detector.h"
#include "stopline_detector.h"
#include "metrics.h"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace filesystem;

int main() {
    TrafficlightDetector::DetectionParams params;
    TrafficlightDetector traffic_light_detector = TrafficlightDetector(params);
    StoplineDetector stop_line_detector = StoplineDetector();
    VehicleDetector vehicle_detector = VehicleDetector("models/yolov5s_clean.onnx");
    ViolationDetector violation_detector = ViolationDetector(traffic_light_detector, vehicle_detector, stop_line_detector);
    Metrics metrics_calculator = Metrics(traffic_light_detector, vehicle_detector, stop_line_detector, violation_detector);

    const string basePath = "Dataset_Traffic_light_project/Label";
    
    for (const auto& folder : directory_iterator(basePath)) {
        if (folder.is_directory()) {
            string pngFile, jsonFile;
            
            for (const auto& file : directory_iterator(folder)) {
                string ext = file.path().extension().string();
                string name = file.path().filename().string();
                
                if (ext == ".png" && name.find("_output") == string::npos)
                    pngFile = file.path().string();
                else if (ext == ".json")
                    jsonFile = file.path().string();
            }
            
            cout << "Folder: " << folder.path().filename() << "\n";
            Mat frame = imread(pngFile);
            
            //read jsonfile 
            ifstream jsonStream(jsonFile);
            stringstream buffer;
            buffer << jsonStream.rdbuf();
            string jsonContent = buffer.str();
            
            Metrics::GroundTruthData data = Metrics::Parse(jsonContent);
            metrics_calculator.ComputeMetricsFrame(frame, data);
        }
    }
    metrics_calculator.ComputeMetrics();

    cout << "Test Video" << endl;
    VideoCapture cap("Dataset_Traffic_light_project/Video_training_set/aziz1.MP4");

    if (!cap.isOpened()) {
        cerr << "cannot open the video" << endl;
        return -1;
    }

    Rect tl_roi = Rect(0, 500, 250, 300); // x, y, width, height
    violation_detector.DetectViolationsonVideo(cap, tl_roi);

    cap.release();

    return 0;
}
