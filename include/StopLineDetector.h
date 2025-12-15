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
    void Preprocessing(const cv::Mat& img, cv::Mat& enh, double cut_ratio);
    void OtsuCanny(const cv::Mat& img, cv::Mat& edges, double low_factor, double high_factor);
    std::vector<cv::Vec4i> FilterHorizontalLines(const std::vector<cv::Vec4i>& lines, double max_angle_deg);
    std::vector<std::vector<cv::Vec4i>> LineClustering(const std::vector<cv::Vec4i>& lines, double max_distance, double max_angle_deg);
    std::vector<cv::Vec4i> FindBestCluster(const std::vector<std::vector<cv::Vec4i>>& clusters, int imgWidth, double min_coverage_rate);
    cv::Vec4i ComputeStopLineEdge(const std::vector<cv::Vec4i>& cluster, int imgWidth);
    double SegmentDistance(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d);
    double distPointSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);
};
#endif // STOP_LINE_DETECTOR_H