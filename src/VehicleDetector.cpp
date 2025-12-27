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
    // Names of the classes found in the Yolov5 which is trained with dataset coco and used these names
    classNames = vector<string>(80);
    classNames[2] = "car";
    classNames[3] = "motorcycle";
    classNames[5] = "bus";
    classNames[7] = "truck";
}

// Class which returns a boolean value if there is or not a Vehicle declared in the loadClassNames
bool VehicleDetector::isVehicle(int classId) {
    return classId == 3 || classId == 2 || classId == 5 || classId == 7;
}


vector<Rect> VehicleDetector::detect(const Mat& frame, Rect roi) {
    Mat frameROI = frame(roi);

    //draw roi bounds on the frame
    rectangle(frame, roi, Scalar(255, 0, 0), 2);

    
    //Mat blob = blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(), true, false);
    Mat blob = blobFromImage(frameROI, 1/255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    vector<Mat> selectedOutput = { outputs[0] };

    return postprocess(frame, outputs, roi.y, frameROI.size());
}

vector<Rect> VehicleDetector::postprocess(const Mat& frame, const vector<Mat>& outs, int roiYOffset, Size roiSize) {
    // outs it should include oly one emement with shape (1, 25200, 85)
    const Mat& output = outs[0]; // (1, 25200, 85)
    const int dimensions = output.size[2]; // 85
    const int rows = output.size[1]; // 25200

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    int inputWidth = 640;
    int inputHeight = 640;
    int roiWidth = roiSize.width;
    int roiHeight = roiSize.height;
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

    //float scale = min(inputWidth / (float)frameWidth, inputHeight / (float)frameHeight);
    //float dx = (inputWidth - frameWidth * scale) / 2.0f;
    //float dy = (inputHeight - frameHeight * scale) / 2.0f;

    float scale = min(inputWidth / (float)roiWidth, inputHeight / (float)roiHeight);
    float dx = (inputWidth - roiWidth * scale) / 2.0f;
    float dy = (inputHeight - roiHeight * scale) / 2.0f;

    const float* data = (float*)output.data;

    for (int i = 0; i < rows; ++i, data += dimensions) {
        float objness = data[4];
        if (objness >= confidenceThreshold) {
            Mat scores(1, dimensions - 5, CV_32FC1, (void*)(data + 5));
            Point classIdPoint;
            double classScore;
            minMaxLoc(scores, 0, &classScore, 0, &classIdPoint);

            float confidence = objness * static_cast<float>(classScore);
            if (confidence > confidenceThreshold && isVehicle(classIdPoint.x)) {
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((cx - w / 2 - dx) / scale);
                int top = static_cast<int>((cy - h / 2 - dy) / scale);
                int width = static_cast<int>(w / scale);
                int height = static_cast<int>(h / scale);

                // limits the boxes inside the image
                //left = max(0, min(left, frameWidth - 1));
                //top = max(0, min(top, frameHeight - 1));

                // Offset to full-frame coordinates
                left = max(0, min(left, roiWidth - 1));
                top = max(0, min(top, roiHeight - 1));

                //if (left + width > frameWidth) width = frameWidth - left;
                //if (top + height > frameHeight) height = frameHeight - top;

                if (left + width > roiWidth) width = roiWidth - left;
                if (top + height > roiHeight) height = roiHeight - top;

                top += roiYOffset;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    vector<Rect> detectedBoxes;

    for (int idx : indices) {
        Rect box = boxes[idx];
        detectedBoxes.push_back(box);
        
        rectangle(frame, box, Scalar(0, 255, 0), 2);

        string label = classNames[classIds[idx]];
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int top = max(box.y, labelSize.height);

        rectangle(frame, Point(box.x, top - labelSize.height),
                      Point(box.x + labelSize.width, top + baseLine),
                      Scalar(0, 255, 0), FILLED);
        putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar(255, 255, 255), 1);
    }
    
    return detectedBoxes;
}