/*
 *  StopLineDetector.cpp
 *  Author: Leonardo Pontello
 */

#include "StopLineDetector.h"

using namespace cv;
using namespace std;

StopLineDetector::StopLineDetector(
    double roiCutRatio,
    double cannyLowFactor,
    double cannyHighFactor,
    int houghThreshold,
    double houghMinLineRatio,
    double houghMaxGapRatio,
    double maxHorizontalAngle,
    double clusterMaxDistance,
    double clusterMaxAngle,
    double minCoverageRate)
    : 
    roiCutRatio_(roiCutRatio),
    cannyLowFactor_(cannyLowFactor),
    cannyHighFactor_(cannyHighFactor),
    houghThreshold_(houghThreshold),
    houghMinLineRatio_(houghMinLineRatio),
    houghMaxGapRatio_(houghMaxGapRatio),
    maxHorizontalAngle_(maxHorizontalAngle),
    clusterMaxDistance_(clusterMaxDistance),
    clusterMaxAngle_(clusterMaxAngle),
    minCoverageRate_(minCoverageRate)
{}

Vec4i StopLineDetector::detectStopLine(const Mat& img)
{
    int roiOffset = static_cast<int>(img.rows * (1.0 - roiCutRatio_));
    
    Mat prep, edges;
    Preprocessing(img, prep, roiCutRatio_);
    OtsuCanny(prep, edges, cannyLowFactor_, cannyHighFactor_);
        
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, houghThreshold_, static_cast<double>(img.cols) * houghMinLineRatio_, static_cast<double>(img.cols) * houghMaxGapRatio_);
    
    // Convert ROI coordinates to original image coordinates
    for (auto& line : lines) {
        line[1] += roiOffset;
        line[3] += roiOffset;
    }

    vector<Vec4i> horizontalLines = FilterHorizontalLines(lines, maxHorizontalAngle_);
    vector<vector<Vec4i>> clusters = LineClustering(horizontalLines, clusterMaxDistance_, clusterMaxAngle_);
    vector<Vec4i> bestCluster = FindBestCluster(clusters, img.cols, minCoverageRate_);
    Vec4i stopLine = ComputeStopLineEdge(bestCluster, img.cols);

    return stopLine;
}

void StopLineDetector::Preprocessing(const Mat& img, Mat& closed, double cut_ratio) 
{
    Mat gray, enh;

    // Cut the region of interest (ROI)
    int cut = static_cast<int>(img.rows * cut_ratio); 
    Rect roi(0, img.rows - cut, img.cols, cut);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray = gray(roi);

    // Histogram Equalization CLAHE
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    clahe->apply(gray, enh);

    // Morphological Closing to enhance horizontal lines
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15,3));
    morphologyEx(enh, closed, MORPH_CLOSE, kernel);
}

void StopLineDetector::OtsuCanny(const Mat& img, Mat& edges, double low_factor, double high_factor) 
{
    Mat blurred, bin;

    GaussianBlur(img, blurred, Size(5, 5), 1.4);

    // Apply Otsu's method to determine Canny thresholds
    double otsuThresh = threshold(blurred, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double lower = max(0.0, low_factor * otsuThresh);
    double upper = min(255.0, high_factor * otsuThresh);

    Canny(blurred, edges, lower, upper);
}

vector<Vec4i> StopLineDetector::FilterHorizontalLines(const vector<Vec4i>& lines, double max_angle_deg) 
{
    // Convert degrees to radians
    double MAX_HORIZONTAL_ANGLE = max_angle_deg * CV_PI / 180.0;
    
    vector<Vec4i> filtered;

    // Filter lines based on their angle
    for (const auto& line : lines) {
        double angle = atan2(line[3] - line[1], line[2] - line[0]);
        
        if (fabs(angle) < MAX_HORIZONTAL_ANGLE || 
            fabs(angle - CV_PI) < MAX_HORIZONTAL_ANGLE) {
            filtered.push_back(line);
        }
    }
    return filtered;
}

vector<vector<Vec4i>> StopLineDetector::LineClustering(const vector<Vec4i>& lines, double max_distance, double max_angle_deg) 
{   
    vector<vector<Vec4i>> clusters;

    double maxAngle = max_angle_deg * CV_PI / 180.0;
    
    // Lambda predicate for line similarity based on distance and angle
    auto predicate = [this, max_distance, maxAngle](const Vec4i& l1, const Vec4i& l2) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        // Calculate delta angle between the two lines
        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI / 2) {
            dAngle = CV_PI - dAngle;
        }
        // Calculate minimum distance between the two line segments
        double dist = SegmentDistance(p1, p2, q1, q2);

        return (dist < max_distance) && (dAngle < maxAngle);
    };

    vector<int> labels;
    // Perform clustering using the partition function
    int nLabels = partition(lines, labels, predicate);
    clusters.resize(nLabels);
    
    for (size_t i = 0; i < lines.size(); ++i) {
        clusters[labels[i]].push_back(lines[i]);
    }

    return clusters;
}

vector<Vec4i> StopLineDetector::FindBestCluster(const vector<vector<Vec4i>>& clusters, int imgWidth, double min_coverage_rate)
{
    // Return empty if no clusters found
    if (clusters.empty()) {
        return {};
    }

    vector<Vec4i> bestCluster;
    double lowestY = 0.0;
    double minCoverage = min_coverage_rate * imgWidth;
    
    for (const auto& cluster : clusters) {
        if (cluster.empty()) continue;
        
        double ySum = 0.0;
        double coverageSum = 0.0;
        
        for (const auto& line : cluster) 
        {
            // Sum of y-coordinates of line midpoints
            ySum += (line[1] + line[3]) / 2.0;
            // Sum of horizontal coverage of the cluster
            coverageSum += fabs(line[2] - line[0]);
        }
        // Check if the cluster meets the minimum coverage requirement
        if (coverageSum < minCoverage) continue;
        // Update the best cluster based on the lowest average y-coordinate (highest y), the lowest line should be the stop line
        if (ySum > lowestY) {
            lowestY = ySum;
            bestCluster = cluster;
        }
    }
    
    return bestCluster;
}

Vec4i StopLineDetector::ComputeStopLineEdge(const vector<Vec4i>& cluster, int imgWidth) 
{
    // Return zero line if the cluster is empty
    if (cluster.empty()) {
        return Vec4i(0, 0, 0, 0);
    }
    
    vector<Point2f> points;
    int minX = INT_MAX, maxX = INT_MIN;
    // Collect all endpoints from the lines in the cluster
    for (const auto& line : cluster) {
        points.emplace_back(line[0], line[1]);
        points.emplace_back(line[2], line[3]);
        minX = min(minX, min(line[0], line[2]));
        maxX = max(maxX, max(line[0], line[2]));
    }
    
    // Fit a line to the collected points using least squares
    Vec4f fittedLine;
    fitLine(points, fittedLine, DIST_L2, 0, 0.01, 0.01);
    
    // fittedLine contains: [vx, vy, x0, y0]
    float vx = fittedLine[0];
    float vy = fittedLine[1];
    float x0 = fittedLine[2];
    float y0 = fittedLine[3];
    
    // Compute y coordinates at image borders   
    float t1 = (minX - x0) / vx;
    int y1 = static_cast<int>(y0 + t1 * vy);
    
    float t2 = (maxX - x0) / vx;
    int y2 = static_cast<int>(y0 + t2 * vy);
    
    return Vec4i(minX, y1, maxX, y2);
}
    
double StopLineDetector::SegmentDistance(const Point2f& a, const Point2f& b, const Point2f& c, const Point2f& d) 
{
    double d1 = distPointSegment(a, c, d);
    double d2 = distPointSegment(b, c, d);
    double d3 = distPointSegment(c, a, b);
    double d4 = distPointSegment(d, a, b);
    return min({d1, d2, d3, d4});
}

double StopLineDetector::distPointSegment(const Point2f& p, const Point2f& a, const Point2f& b) 
{
    Point2f ab = b - a;
    double abLengthSq = ab.dot(ab);
    
    // if a and b are the same point
    if (abLengthSq < 1e-10) {
        return norm(p - a);
    }
    
    Point2f ap = p - a;
    double t = clamp(ap.dot(ab) / abLengthSq, 0.0, 1.0);
    Point2f proj = a + t * ab;
    
    return norm(p - proj);
}