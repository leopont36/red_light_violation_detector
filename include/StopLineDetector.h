#ifndef STOPLINEDETECTOR_H
#define STOPLINEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class StopLineDetector 
{
public:
    StopLineDetector();
    
    
    cv::Vec4i detectStopLine(const cv::Mat& img) const;

private:
    
    void Preprocessing(const cv::Mat& img, cv::Mat& closed, double cut_ratio) const;
    
    void OtsuCanny(const cv::Mat& img, cv::Mat& edges, double low_factor = 0.5, double high_factor = 1.5) const;
    
    std::vector<cv::Vec4i> FilterHorizontalLines(const std::vector<cv::Vec4i>& lines, double max_angle_deg = 15.0) const;
    
    std::vector<std::vector<cv::Vec4i>> LineClustering(const std::vector<cv::Vec4i>& lines, double max_distance = 5.0, double max_angle_deg = 5.0) const;
    
    std::vector<cv::Vec4i> FindBestCluster(const std::vector<std::vector<cv::Vec4i>>& clusters, int imgWidth, double min_coverage_rate = 0.40) const;
    
    cv::Vec4i ComputeStopLineEdge(const std::vector<cv::Vec4i>& cluster, int imgWidth) const;
    
    double SegmentDistance(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d) const;
    
    double distPointSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) const;
};

#endif // STOPLINEDETECTOR_H