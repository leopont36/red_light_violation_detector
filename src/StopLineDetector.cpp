#include "StopLineDetector.h"

using namespace cv;
using namespace std;

std::function<bool(const cv::Vec4i&, const cv::Vec4i&)> StopLineDetector::linePredicate_;

StopLineDetector::StopLineDetector() 
{ 
    linePredicate_ = [this](const cv::Vec4i& l1, const cv::Vec4i& l2) 
    {
        cv::Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        cv::Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI / 2) dAngle = CV_PI - dAngle;

        double dist = segmentDistance(p1, p2, q1, q2);

        double maxDist = 20.0;
        double maxAngle = CV_PI / 36.0;

        return (dist < maxDist) && (dAngle < maxAngle);
    };
}

Vec4i StopLineDetector::detectStopLine(const Mat& img) 
{
    Mat prep, edges;
    Preprocessing(img, prep);
    OtsuCanny(prep, edges);
        
    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    std::vector<Vec4i> horizontalLines;
    for (const auto& line : lines) 
    {
        Point2f p1(line[0], line[1]), p2(line[2], line[3]);
        double angle = atan2(p2.y - p1.y, p2.x - p1.x);
        if (fabs(angle) < CV_PI/36.0 || fabs(fabs(angle) - CV_PI) < CV_PI/36.0) 
            horizontalLines.push_back(line);
    }

    vector<int> labels;
    int nLabels = partition(horizontalLines, labels, linePredicate_);

    // a sto punto ho tante linee orizzontali, devo trovare l'algoritmo per segliere quella giusta 

    return Vec4i(); // Placeholder, replace with logic to find the best stop line
    
}

void StopLineDetector::Preprocessing(const Mat& img, Mat& enh) 
{
    Mat gray;
    int cut = img.rows * 0.30; 
    Rect roi(0, img.rows - cut, img.cols, cut);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray = gray(roi);
    Ptr<CLAHE> clahe = createCLAHE(2.0 , Size(8,8));
    clahe->apply(gray, enh);    
}

void StopLineDetector::OtsuCanny(const Mat& img, Mat& edges)
{
    Mat closed, blurred, bin;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 3));
    morphologyEx(img, closed, MORPH_CLOSE, kernel);

    GaussianBlur(closed, blurred, Size(5,5), 1.4);

    double otsuThresh = threshold(blurred, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double lower = max(0.0, 0.5 * otsuThresh);
    double upper = min(255.0, 1.5 * otsuThresh);
    Canny(closed, edges, lower, upper);
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
    
double StopLineDetector::segmentDistance(const Point2f& a, const Point2f& b, const Point2f& c, const Point2f& d) 
{
    double d1 = distPointSegment(a, c, d);
    double d2 = distPointSegment(b, c, d);
    double d3 = distPointSegment(c, a, b);
    double d4 = distPointSegment(d, a, b);
    return std::min({d1, d2, d3, d4});
}
