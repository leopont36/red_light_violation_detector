/*
 *  VehicleDetector.cpp
 *  Author: Angelica Zonta
 */

#include "VehicleDetector.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

// Constructor
VehicleDetector::VehicleDetector(const string& modelPath, float confThreshold, float nmsThreshold)
    : confidenceThreshold(confThreshold), nmsThreshold(nmsThreshold)
{
    net = readNetFromONNX(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    loadClassNames();
}

void VehicleDetector::loadClassNames() {
    // Only store the vehicle classes we actually use (YOLOv5 COCO indices)
    classNames = {
        {2, "car"},
        {3, "motorcycle"},
        {5, "bus"},
        {7, "truck"}
    };
}

// Returns true if the class ID corresponds to a vehicle
bool VehicleDetector::isVehicle(int classId) {
    return classNames.count(classId) > 0;
}

vector<Rect> VehicleDetector::detect(const Mat& frame, Rect roi) {
    Mat frameROI = frame(roi);
    
    if(roi.width > 0 && roi.height > 0)
        frameROI = frame(roi);
    else
        frameROI = frame;

    Mat blob = blobFromImage(frameROI, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return postprocess(frame, outputs, roi.y, frameROI.size());
}

vector<Rect> VehicleDetector::postprocess(const Mat& frame, const vector<Mat>& outs, int roiYOffset, Size roiSize) {
    // outs[0] has shape (1, 25200, 85)
    const Mat& output = outs[0];
    const int dimensions = output.size[2]; // 85
    const int rows = output.size[1]; // 25200
    const int numClasses = dimensions - 5;

    const int inputWidth = 640;
    const int inputHeight = 640;
    const int roiWidth = roiSize.width;
    const int roiHeight = roiSize.height;

    float scale = min(inputWidth / (float)roiWidth, inputHeight / (float)roiHeight);
    float dx = (inputWidth - roiWidth  * scale) / 2.0f;
    float dy = (inputHeight - roiHeight * scale) / 2.0f;

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    const float* data = (float*)output.data;

    for (int i = 0; i < rows; ++i, data += dimensions) {
        float objness = data[4];
        if (objness < confidenceThreshold)
            continue;

        const float* scores = data + 5;
        int bestClass = (int)(max_element(scores, scores + numClasses) - scores);
        float classScore = scores[bestClass];

        float confidence = objness * classScore;
        if (confidence <= confidenceThreshold || !isVehicle(bestClass))
            continue;

        float cx = data[0], cy = data[1];
        float w = data[2], h = data[3];

        int left = static_cast<int>((cx - w / 2 - dx) / scale);
        int top = static_cast<int>((cy - h / 2 - dy) / scale);
        int width = static_cast<int>(w / scale);
        int height = static_cast<int>(h / scale);

        left = max(0, min(left, roiWidth  - 1));
        top  = max(0, min(top,  roiHeight - 1));

        if (left + width > roiWidth) 
            width = roiWidth - left;
        if (top + height > roiHeight) 
            height = roiHeight - top;
        top += roiYOffset;

        classIds.push_back(bestClass);
        confidences.push_back(confidence);
        boxes.push_back(Rect(left, top, width, height));
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    vector<Rect> detectedBoxes;
    detectedBoxes.reserve(indices.size());

    for (int idx : indices) {
        Rect box = boxes[idx];
        detectedBoxes.push_back(box);

        // [DEBUG] rendering — ideally move this outside postprocess
        rectangle(frame, box, Scalar(0, 255, 0), 2);

        const string& label = classNames.at(classIds[idx]);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int labelTop = max(box.y, labelSize.height);

        rectangle(frame, Point(box.x, labelTop - labelSize.height),
            Point(box.x + labelSize.width, labelTop + baseLine),
            Scalar(0, 255, 0), FILLED);

        putText(frame, label, Point(box.x, labelTop), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    }

    return detectedBoxes;
}