/*
 *  StopLineDetector.h
 *  Author: Leonardo Pontello
 */

#ifndef STOP_LINE_DETECTOR_H
#define STOP_LINE_DETECTOR_H

#include <opencv2/opencv.hpp>

class StopLineDetector
{
public:
    StopLineDetector();
    cv::Vec4i detectStopLine(const cv::Mat& image);
private:
    cv::Mat Preprocessing(const cv::Mat& img);
    double segmentDistance(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d);
    double distPointSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);
    static std::function<bool(const cv::Vec4i&, const cv::Vec4i&)> linePredicate_;
};
#endif // STOP_LINE_DETECTOR_H
