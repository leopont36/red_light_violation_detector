#include "StopLineDetector.h"

using namespace cv;
using namespace std;


StopLineDetector::StopLineDetector(){}

Vec4i StopLineDetector::detectStopLine(const Mat& img) 
{
    const double ROI_CUT_RATIO = 0.70;

    int roiOffset = static_cast<int>(img.rows * (1.0 - ROI_CUT_RATIO));
    
    Mat prep, edges;
    Preprocessing(img, prep, ROI_CUT_RATIO);
    OtsuCanny(prep, edges, 0.5, 1.5);
        
    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, static_cast<double>(img.cols) * 0.20, static_cast<double>(img.cols) * 0.10);
    
    // Convert ROI coordinates to original image coordinates
    for (auto& line : lines) {
        line[1] += roiOffset;
        line[3] += roiOffset;
    }

    std::vector<Vec4i> horizontalLines = FilterHorizontalLines(lines, 15.0);

    vector<vector<Vec4i>> clusters = LineClustering(horizontalLines, 5.0, 5.0);

    vector<Vec4i> bestCluster = FindBestCluster(clusters, img.cols, 0.40);

    // MODIFICA: Usa la nuova funzione che mantiene l'inclinazione
    Vec4i stopLine = ComputeStopLineEdge(bestCluster, img.cols);

    return stopLine;
}

void StopLineDetector::Preprocessing(const Mat& img, Mat& closed, double cut_ratio) 
{
    Mat gray, enh;

    // Cut the region of interest (ROI): 70% of the bottom part of the image 
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

void StopLineDetector::OtsuCanny(const Mat& img, Mat& edges, double low_factor = 0.5, double high_factor = 1.5)
{
    Mat blurred, bin;

    GaussianBlur(img, blurred, Size(5,5), 1.4);

    // Apply Otsu's method to determine Canny thresholds
    double otsuThresh = threshold(blurred, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double lower = max(0.0, low_factor * otsuThresh);
    double upper = min(255.0, high_factor * otsuThresh);

    Canny(blurred, edges, lower, upper);
}

vector<Vec4i> StopLineDetector::FilterHorizontalLines(const vector<Vec4i>& lines, double max_angle_deg = 15.0) 
{
    const double MAX_HORIZONTAL_ANGLE = max_angle_deg * CV_PI / 180.0; // radians convertion
    
    vector<Vec4i> filtered;
    
    for (const auto& line : lines) {
        Point2f p1(line[0], line[1]), p2(line[2], line[3]);
        double angle = atan2(line[3] - line[1], line[2] - line[0]);
        
        if (fabs(angle) < MAX_HORIZONTAL_ANGLE || fabs(angle - CV_PI) < MAX_HORIZONTAL_ANGLE) 
            filtered.push_back(line);    
    }
    return filtered;
}

vector<vector<Vec4i>> StopLineDetector::LineClustering(const vector<Vec4i>& lines, double max_distance = 5.0, double max_angle_deg = 5.0) 
{   
    vector<vector<Vec4i>> clusters;
    
    auto predicate = [this](const Vec4i& l1, const Vec4i& l2) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI/2) dAngle = CV_PI - dAngle;

        double dist = SegmentDistance(p1, p2, q1, q2);
        double maxAngle = max_angle_deg * CV_PI / 180.0;

        return (dist < max_distance) && (dAngle < maxAngle);
    };

    vector<int> labels;
    int nLabels = partition(lines, labels, predicate);
    clusters.resize(nLabels);
    
    for (size_t i = 0; i < lines.size(); ++i) {
        clusters[labels[i]].push_back(lines[i]);
    }

    return clusters;
}

vector<Vec4i> StopLineDetector::FindBestCluster(const vector<vector<Vec4i>>& clusters,  int imgWidth, double min_coverage_rate = 0.40) 
{
    vector<Vec4i> bestCluster;
    double lowestY = 0.0;
    
    for (const auto& cluster : clusters) {
        if (cluster.empty()) continue;
        
        double lengthSum = 0.0;
        double ySum = 0.0;
        double coverageSum = 0.0;
        
        for (const auto& line : cluster) {
            ySum += (line[1] + line[3]) / 2.0;
            coverageSum += fabs(line[2] - line[0]);
        }
        
        if (coverageSum < min_coverage_rate * imgWidth) 
            continue;
        
        if (ySum > lowestY) {
            lowestY = ySum;
            bestCluster = cluster;
        }
    }
    
    return bestCluster;
}

// NUOVA FUNZIONE: Calcola linea rappresentativa mantenendo l'inclinazione
Vec4i StopLineDetector::ComputeStopLineEdge(const vector<Vec4i>& cluster, int imgWidth) 
{
    if (cluster.empty()) return Vec4i(0, 0, 0, 0);
    
    // Raccoglie tutti i punti del cluster
    vector<Point2f> points;
    int minX = INT_MAX, maxX = INT_MIN;
    
    for (const auto& line : cluster) {
        points.push_back(Point2f(line[0], line[1]));
        points.push_back(Point2f(line[2], line[3]));
        minX = min(minX, min(line[0], line[2]));
        maxX = max(maxX, max(line[0], line[2]));
    }
    
    // Fit di una linea con least squares
    Vec4f fittedLine;
    fitLine(points, fittedLine, DIST_L2, 0, 0.01, 0.01);
    
    // fittedLine contiene: [vx, vy, x0, y0]
    // dove (vx, vy) è la direzione e (x0, y0) è un punto sulla linea
    float vx = fittedLine[0];
    float vy = fittedLine[1];
    float x0 = fittedLine[2];
    float y0 = fittedLine[3];
    
    // Calcola i punti estremi sulla linea fittata
    // Parametric form: (x, y) = (x0, y0) + t * (vx, vy)
    // Per x = minX: t = (minX - x0) / vx
    float t1 = (minX - x0) / vx;
    int y1 = static_cast<int>(y0 + t1 * vy);
    
    // Per x = maxX: t = (maxX - x0) / vx
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
    return std::min({d1, d2, d3, d4});
}

double StopLineDetector::distPointSegment(const Point2f& p, const Point2f& a, const Point2f& b) 
{
    Point2f ab = b - a;
    Point2f ap = p - a;
    double t = (ap.dot(ab)) / (ab.dot(ab) + 1e-9);
    t = std::max(0.0, std::min(1.0, t));
    Point2f proj = a + t * ab;
    return norm(p - proj);
}