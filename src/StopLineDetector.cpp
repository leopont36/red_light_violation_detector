#include "StopLineDetector.h"

using namespace cv;
using namespace std;

std::function<bool(const cv::Vec4i&, const cv::Vec4i&)> StopLineDetector::linePredicate_;

StopLineDetector::StopLineDetector() 
{ 
}

Vec4i StopLineDetector::detectStopLine(const Mat& img) 
{
    Mat prep, edges;
    Preprocessing(img, prep);
    OtsuCanny(prep, edges);
        
    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, img.cols * 0.20, img.cols * 0.10);
    
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
    const double ROI_CUT_RATIO = 0.30;
    const double CLAHE_CLIP_LIMIT = 2.0;
    const Size CLAHE_TILE_SIZE = Size(8,8);  
    Mat gray;

    //cut the region of interest (ROI): 70% of the bottom part of the image 
    int cut = img.rows * ROI_CUT_RATIO; 
    Rect roi(0, img.rows - cut, img.cols, cut);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray = gray(roi);

    // Histogram Equalization CLAHE
    Ptr<CLAHE> clahe = createCLAHE(CLAHE_CLIP_LIMIT , CLAHE_TILE_SIZE);
    clahe->apply(gray, enh);    
}

void StopLineDetector::OtsuCanny(const Mat& img, Mat& edges)
{
    const Size KERNEL_SIZE = Size(15,3);
    const Size BLUR_SIZE = Size(5,5);
    const double SIGMA = 1.4;
    Mat closed, blurred, bin;

    Mat kernel = getStructuringElement(MORPH_RECT, KERNEL_SIZE);
    morphologyEx(img, closed, MORPH_CLOSE, kernel);

    GaussianBlur(closed, blurred, BLUR_SIZE, SIGMA);

    double otsuThresh = threshold(blurred, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    double lower = max(0.0, 0.5 * otsuThresh);
    double upper = min(255.0, 1.5 * otsuThresh);
    Canny(closed, edges, lower, upper);
}

vector<Vec4i> filterLines(const vector<Vec4i>& lines, const Mat& img, const Rect& roi) {
    vector<Vec4i> filtered;
    double maxAngle = params.maxHorizontalAngleDeg * CV_PI / 180.0;
    
    for (auto &line : lines) {
        // Trasla coordinate rispetto a ROI
        Vec4i globalLine = line;
        globalLine[1] += roi.y;
        globalLine[3] += roi.y;
        
        LineMetrics metrics(globalLine, img.cols, img.rows);
        
        // Filtri multipli
        bool isHorizontal = (fabs(metrics.angle) < maxAngle || 
                            fabs(fabs(metrics.angle) - CV_PI) < maxAngle);
        bool hasGoodCoverage = metrics.horizontalCoverage > params.minHorizontalCoverage;
        bool isInLowerPart = metrics.avgY > img.rows * params.roiLowerThirdOnly;
        
        if (isHorizontal && hasGoodCoverage && isInLowerPart) {
            filtered.push_back(globalLine);
        }
    }
    
    cout << "Linee filtrate: " << filtered.size() << " / " << lines.size() << endl;
    return filtered;
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
