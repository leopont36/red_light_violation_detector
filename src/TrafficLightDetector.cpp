/*
 *  TrafficLightDetector.cpp
 *  Author: Milica Masic
 */
 
#include "TrafficLightDetector.h"
#include <iostream>

using namespace cv;
using namespace std;

TrafficLightDetector::TrafficLightDetector(const DetectionParams& params)
    : params_(params) {}

void TrafficLightDetector::detectAndAnnotate(Mat& img) {
    if (img.empty()) {
        cout << "Error: empty image passed to detector" << endl;
        return;
    }

    //debug
    //cout << "Image size: " << img.cols << "x" << img.rows << endl;

    Mat cropped = img(params_.roi);

    //debug
    rectangle(img, params_.roi, cv::Scalar(0, 255, 0), 2);

    //preprocessing
    Mat grayscale, blurred;
    cvtColor(cropped, grayscale, COLOR_BGR2GRAY);
    GaussianBlur(grayscale, blurred, Size(9, 9), 2);

    //circle detection
    vector<Vec3f> circles;

    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1,
                cropped.rows / 8,
                params_.houghParam1,
                params_.houghParam2,
                params_.minRadius,
                params_.maxRadius);

    for (int i = 0; i < circles.size(); i++) {
        Vec3f c = circles[i];
        Point center(cvRound(c[0]) + params_.roi.x, cvRound(c[1]) + params_.roi.y);
        int radius = cvRound(c[2]);

        //debug
        //cout << "Radius[i]: " << radius << endl;

        //detect color in the patches
        int patchSize = 3;
        Rect patch(center.x - patchSize, center.y - patchSize, patchSize * 2, patchSize * 2);
        patch.x = max(patch.x, 0);
        patch.y = max(patch.y, 0);
        patch.width = min(patch.width, img.cols - patch.x);
        patch.height = min(patch.height, img.rows - patch.y);

        Mat patchROI = img(patch);

    
        string color = getColorFromPatch(patchROI, patch, img);

        //draw the circles on the frame
        circle(img, center, radius, Scalar(0, 255, 0), 2);

        //debug
        cout << "Detected traffic light color: " << color << endl;

        //write the color in the image
        putText(img, color, Point(center.x - 10, center.y - radius - 10),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    }
}

string TrafficLightDetector::getColorFromPatch(const Mat& patch, const Rect& patchRect, const Mat& img) {
    Mat hsvPatch;
    cvtColor(patch, hsvPatch, COLOR_BGR2HSV);
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);

    int medianHue = getMedianHueWithFallback(hsvPatch, patchRect, imgHSV);
    //cout << "H: " << medianHue << endl;
    if (medianHue == -1) return "";

    if ((medianHue < 15 || medianHue > 160))
        return "RED";
    else if (medianHue >= 20 && medianHue <= 35)
        return "YELLOW";
    else if (medianHue >= 40 && medianHue <= 85)
        return "GREEN";

    return "";
}

int TrafficLightDetector::getMedianHueWithFallback(const Mat& hsv, const Rect& patchRect, const Mat& imgHSV) {
    vector<int> validHues;
    const int S_THRESH = 50; 
    const int V_OVEREXPOSURE_THRESH = 240;
    const int V_UNDEREXPOSURE_THRESH = 50;

    int total = 0;
    int underexposedCount = 0;
    int overexposedCount = 0;

    //1. filter inner patch
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            Vec3b hsvPixel = hsv.at<Vec3b>(y, x);
            int h = hsvPixel[0], s = hsvPixel[1], v = hsvPixel[2];

            total ++;

            if (v < V_UNDEREXPOSURE_THRESH) {
                underexposedCount++;
                continue;
            }
            if (v > V_OVEREXPOSURE_THRESH || s < S_THRESH) {
                overexposedCount++;
                continue;
            }

            validHues.push_back(h);
        }
    }

    //2. if valid, return median
    if (!validHues.empty()) {
        sort(validHues.begin(), validHues.end());
        return validHues[validHues.size() / 2];
    }

    //3. handle if circle fully underexposed or overexposed
    if (underexposedCount == total) {
        //too dark — no fallback
        return -1;
    }

    //4. check ring pixels around - red glow

    if (overexposedCount == total) {
        validHues.clear();
        int ringThickness = 20; //how far to look outside
        int startX = max(patchRect.x - ringThickness, 0);
        int startY = max(patchRect.y - ringThickness, 0);
        int endX = min(patchRect.x + patchRect.width + ringThickness, imgHSV.cols);
        int endY = min(patchRect.y + patchRect.height + ringThickness, imgHSV.rows);

        for (int y = startY; y < endY; ++y) {
            for (int x = startX; x < endX; ++x) {
                //skip inner region 
                if (x >= patchRect.x && x < patchRect.x + patchRect.width &&
                    y >= patchRect.y && y < patchRect.y + patchRect.height)
                    continue;

                Vec3b hsvPixel = imgHSV.at<Vec3b>(y, x);
                int h = hsvPixel[0], s = hsvPixel[1], v = hsvPixel[2];

                if (v > V_UNDEREXPOSURE_THRESH && v < V_OVEREXPOSURE_THRESH && s > S_THRESH) {
                    validHues.push_back(h);
                }
            }
        }

        if (!validHues.empty()) {
            sort(validHues.begin(), validHues.end());
            return validHues[validHues.size() / 2];
        }

        return -1;  //no valid outer pixels

    }
    

    //5. nothing found
    return -1; 
}

