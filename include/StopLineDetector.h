/*
 *  StopLineDetector.h
 *  Author: Leonardo Pontello
 */

#ifndef STOPLINEDETECTOR_H
#define STOPLINEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class StopLineDetector {
private:
    // ROI parameters
    double roiCutRatio_;
    
    // Canny parameters
    double cannyLowFactor_;
    double cannyHighFactor_;
    
    // Hough parameters
    int houghThreshold_;
    double houghMinLineRatio_;
    double houghMaxGapRatio_;
    
    // Filtering parameters
    double maxHorizontalAngle_;
    
    // Clustering parameters
    double clusterMaxDistance_;
    double clusterMaxAngle_;
    
    // Selection parameters
    double minCoverageRate_;

    // Private methods
    void Preprocessing(const cv::Mat& img, cv::Mat& closed, double cut_ratio);
    void OtsuCanny(const cv::Mat& img, cv::Mat& edges, double low_factor, double high_factor);
    std::vector<cv::Vec4i> FilterHorizontalLines(const std::vector<cv::Vec4i>& lines, double max_angle_deg);
    std::vector<std::vector<cv::Vec4i>> LineClustering(const std::vector<cv::Vec4i>& lines, double max_distance, double max_angle_deg);
    std::vector<cv::Vec4i> FindBestCluster(const std::vector<std::vector<cv::Vec4i>>& clusters, int imgWidth, double min_coverage_rate);
    cv::Vec4i ComputeStopLineEdge(const std::vector<cv::Vec4i>& cluster, int imgWidth);
    double SegmentDistance(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d);
    double distPointSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);

public:
    // Constructor with default parameters
    StopLineDetector(
        double roiCutRatio = 0.70,
        double cannyLowFactor = 0.5,
        double cannyHighFactor = 1.5,
        int houghThreshold = 50,
        double houghMinLineRatio = 0.20,
        double houghMaxGapRatio = 0.10,
        double maxHorizontalAngle = 15.0,
        double clusterMaxDistance = 5.0,
        double clusterMaxAngle = 5.0,
        double minCoverageRate = 0.40
    );
    
    cv::Vec4i detectStopLine(const cv::Mat& img);
};

#endif // STOPLINEDETECTOR_H