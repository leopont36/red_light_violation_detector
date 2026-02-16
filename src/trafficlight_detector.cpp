/*
 *  TrafficlightDetector.cpp
 *  Author: Milica Masic
 */

#include "trafficlight_detector.h"
#include <iostream>

using namespace cv;
using namespace std;

TrafficlightDetector::TrafficlightDetector(const DetectionParams& params) : params_(params) {}

void TrafficlightDetector::SetParams(const DetectionParams& params) {
    params_ = params;
}

TrafficlightColor TrafficlightDetector::DetectTrafficlight(const Mat& img, const Rect& roi) {
    if (img.empty()) {
        cout << "Error: empty image passed to detector" << endl;
        return TrafficlightColor::Unknown;
    }

    Mat cropped = img(roi);

    // preprocessing
    Mat grayscale, blurred;
    cvtColor(cropped, grayscale, COLOR_BGR2GRAY);
    GaussianBlur(grayscale, blurred, Size(9, 9), 2);

    // circle detection
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1,
        cropped.rows / 8,
        params_.houghParam1,
        params_.houghParam2,
        params_.minRadius,
        params_.maxRadius);

    // early exit if no circles found
    if (circles.empty())
        return TrafficlightColor::Unknown;

    // convert full image to HSV
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);

    // color voting
    int redVotes = 0, yellowVotes = 0, greenVotes = 0;

    for (const auto& c : circles) {
        Point center(cvRound(c[0]) + roi.x, cvRound(c[1]) + roi.y);

        int patchSize = 3;
        Rect patch(center.x - patchSize, center.y - patchSize, patchSize * 2, patchSize * 2);
        patch.x = max(patch.x, 0);
        patch.y = max(patch.y, 0);
        patch.width = min(patch.width,  img.cols - patch.x);
        patch.height = min(patch.height, img.rows - patch.y);

        Mat patchROI = img(patch);
        TrafficlightColor color = getColorFromPatch(patchROI, patch, imgHSV);

        if (color == TrafficlightColor::Red)    
            redVotes++;
        
        else if (color == TrafficlightColor::Yellow) 
            yellowVotes++;
        
        else if (color == TrafficlightColor::Green)  
            greenVotes++;
    }

    // decide final color based on votes
    if (redVotes > yellowVotes && redVotes > greenVotes && redVotes > 0) 
        return TrafficlightColor::Red;

    if (greenVotes > yellowVotes && greenVotes > redVotes && greenVotes > 0) 
        return TrafficlightColor::Green;

    if (yellowVotes > 0)                                                          
        return TrafficlightColor::Yellow;

    return TrafficlightColor::Unknown;
}

TrafficlightColor TrafficlightDetector::getColorFromPatch(const Mat& patch, const Rect& patchRect, const Mat& imgHSV) {
    
    Mat hsvPatch;
    cvtColor(patch, hsvPatch, COLOR_BGR2HSV);

    int medianHue = getMedianHueWithFallback(hsvPatch, patchRect, imgHSV);
    if (medianHue == -1)
        return TrafficlightColor::Unknown;

    if (medianHue < 10  || medianHue > 170)              
        return TrafficlightColor::Red;
    
    else if (medianHue >= 15 && medianHue <= 40)           
        return TrafficlightColor::Yellow;
    
    else if (medianHue >= 45 && medianHue <= 90)           
        return TrafficlightColor::Green;

    return TrafficlightColor::Unknown;
}

int TrafficlightDetector::getMedianHueWithFallback(const Mat& hsv, const Rect& patchRect, const Mat& imgHSV) {
    vector<int> validHues;
    const int S_THRESH = 40;
    const int V_OVEREXPOSURE_THRESH = 250;
    const int V_UNDEREXPOSURE_THRESH = 40;

    int total = 0;
    int underexposedCount = 0;
    int overexposedCount = 0;

    // 1. filter inner patch
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            Vec3b hsvPixel = hsv.at<Vec3b>(y, x);
            int h = hsvPixel[0], s = hsvPixel[1], v = hsvPixel[2];

            total++;

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

    // 2. if valid, return median
    if (!validHues.empty()) {
        sort(validHues.begin(), validHues.end());
        return validHues[validHues.size() / 2];
    }

    // 3. fully underexposed — no fallback
    if (underexposedCount == total)
        return -1;

    // 4. fully overexposed — check ring pixels around for red glow
    if (overexposedCount == total) {
        validHues.clear();
        int ringThickness = 20;
        int startX = max(patchRect.x - ringThickness, 0);
        int startY = max(patchRect.y - ringThickness, 0);
        int endX = min(patchRect.x + patchRect.width  + ringThickness, imgHSV.cols);
        int endY = min(patchRect.y + patchRect.height + ringThickness, imgHSV.rows);

        for (int y = startY; y < endY; ++y) {
            for (int x = startX; x < endX; ++x) {
                // skip inner region
                if (x >= patchRect.x && x < patchRect.x + patchRect.width &&
                    y >= patchRect.y && y < patchRect.y + patchRect.height)
                    continue;

                Vec3b hsvPixel = imgHSV.at<Vec3b>(y, x);
                int h = hsvPixel[0], s = hsvPixel[1], v = hsvPixel[2];

                if (v > V_UNDEREXPOSURE_THRESH && v < V_OVEREXPOSURE_THRESH && s > S_THRESH)
                    validHues.push_back(h);
            }
        }

        if (!validHues.empty()) {
            sort(validHues.begin(), validHues.end());
            return validHues[validHues.size() / 2];
        }

        return -1; // no valid outer pixels
    }

    // 5. nothing found
    return -1;
}