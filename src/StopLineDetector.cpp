#include "StopLineDetector.h"

using namespace cv;
using namespace std;

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

Vec4i StopLineDetector::detectStopLine(const Mat& image) 
{
    Mat preprocessed = Preprocessing(image);
    
    Mat blurred, edges;
    GaussianBlur(preprocessed, blurred, Size(5,5), 1.4);
    Canny(blurred, edges, 50, 150);
        
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


Mat StopLineDetector::Preprocessing(const Mat& img) 
{
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);
    std::vector<Mat> lab_planes(3);
    split(lab, lab_planes);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    Mat l_eq;
    clahe->apply(lab_planes[0], l_eq);
    lab_planes[0] = l_eq;

    Mat lab_eq, img_eq;
    merge(lab_planes, lab_eq);
    cvtColor(lab_eq, img_eq, COLOR_Lab2BGR);

    Mat gray;
    cvtColor(img_eq, gray, COLOR_BGR2GRAY);

    Mat background;
    GaussianBlur(gray, background, Size(51, 51), 0);

    Mat diff;
    absdiff(gray, background, diff);

    Mat enhanced;
    diff.convertTo(enhanced, -1, 2.0, 0);

    return enhanced;
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




/*

class StopLineDetector 
{
public:
    Vec4i detectStopLine_circles(const Mat& frame) 
    {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat blurred;
        GaussianBlur(gray, blurred, Size(31, 31), 0);

        Mat normalized;
        absdiff(gray, blurred, normalized);

        Mat binary;
        threshold(normalized, binary, 30, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat ellipsesAndTrapezoides = Mat::zeros(binary.size(), CV_8UC1);
        vector<Point2f> ellipseCenters;

        for (const auto& contour : contours) 
        {
            double area = contourArea(contour);
            if (area < 20)
                continue;

            if (contour.size() >= 5) 
            {
                RotatedRect figure = fitEllipse(contour);
                if (figure.size.width > 3 && figure.size.height > 3 && 
                    figure.size.width < 60 && figure.size.height < 60) 
                {
                    ellipse(ellipsesAndTrapezoides, figure, 255, -1);
                    ellipseCenters.push_back(figure.center);
                    continue;
                }
            }
        }

        vector<Vec4i> lines;

        // TODO: make Gaussian blur and Canny thresholds adaptive

        vector<Vec4i> horizontalLines;
        for (const auto& line : lines) 
        {
            if (abs(atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI) < 15.0) 
                horizontalLines.push_back(line);
        }

        // Nota: la funzione non restituisce nulla al momento (Vec4i)
        return Vec4i(); // placeholder, da sostituire con logica vera
    }




    private:

};

/*void on_trackbar(int, void*) {
    if (soglia_alta < soglia_bassa + 1) {
        soglia_alta = soglia_bassa + 1;
        setTrackbarPos("Soglia Alta", "Canny Edge Detector", soglia_alta);
    }
    GaussianBlur(grayImage, blurred, Size(5,5), 0);

    Mat edges;
    Canny(blurred, edges, soglia_bassa, soglia_alta);

    imshow("Canny Edge Detector", edges);
}*/
/*#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

struct LineInfo {
    Vec4i line;
    double length;
    double angle;
    int avgY;
    double score;

    LineInfo(const Vec4i& l) : line(l) {
        length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
        angle = abs(atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI);
        avgY = (l[1] + l[3]) / 2;
        score = length + 0.5 * avgY;
    }

    bool isHorizontal(double tolerance = 15.0) const {
        return (angle < tolerance) || (angle > (180.0 - tolerance));
    }

    string direction() const {
        if (line[3] < line[1]) return "↑";
        else if (line[3] > line[1]) return "↓";
        else return "→";
    }
};

class StopLineDetector {
private:
    struct Parameters {
        Size gaussianKernel = Size(3, 3);
        double gaussianSigma = 0.0;
        double cannyLow = 50.0;
        double cannyHigh = 150.0;
        int houghThreshold = 150;
        double houghMinLineLength = 700.0;
        double houghMaxLineGap = 200.0;
        double horizontalTolerance = 15.0;
        double minLineLength = 80.0;
        int maxIntersections = 3;
    } params;

public:
    Mat processImage(const string& imagePath) {
        Mat img = imread(imagePath, IMREAD_COLOR);
        if (img.empty()) throw runtime_error("Errore: Impossibile caricare l'immagine da: " + imagePath);
        cout << "Immagine caricata: " << img.cols << "x" << img.rows << " pixels" << endl;

        Mat blurred = applyGaussianBlur(img);
        Mat edges = detectEdges(blurred);
        vector<Vec4i> allLines = detectLines(edges);
        vector<LineInfo> horizontalLines = filterHorizontalLines(allLines);

        cout << "Linee totali rilevate: " << allLines.size() << endl;
        cout << "Linee orizzontali filtrate: " << horizontalLines.size() << endl;

        vector<LineInfo> filteredLines = removeHighlyIntersectingLines(horizontalLines, params.maxIntersections);
        cout << "Linee orizzontali dopo filtro intersezioni: " << filteredLines.size() << endl;

        LineInfo stopLine = findBestStopLine(filteredLines);
        displayResults(img, edges, allLines, horizontalLines, filteredLines, stopLine);
        printDetailedInfo(allLines, filteredLines, stopLine);
        return img;
    }

private:
    Mat applyGaussianBlur(const Mat& img) {
        Mat blurred;
        GaussianBlur(img, blurred, params.gaussianKernel, params.gaussianSigma);
        return blurred;
    }

    Mat detectEdges(const Mat& blurred) {
        Mat edges;
        Canny(blurred, edges, params.cannyLow, params.cannyHigh);
        return edges;
    }

    vector<Vec4i> detectLines(const Mat& edges) {
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, params.houghThreshold,
                    params.houghMinLineLength, params.houghMaxLineGap);
        return lines;
    }

    vector<LineInfo> filterHorizontalLines(const vector<Vec4i>& allLines) {
        vector<LineInfo> horizontalLines;
        for (const auto& line : allLines) {
            LineInfo lineInfo(line);
            if (lineInfo.isHorizontal(params.horizontalTolerance) ) {
                horizontalLines.push_back(lineInfo);
            }
        }
        return horizontalLines;
    }

    vector<LineInfo> removeHighlyIntersectingLines(const vector<LineInfo>& lines, int maxIntersections) {
        vector<int> intersectionCounts(lines.size(), 0);
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                Point2f pt;
                if (getIntersectionPoint(lines[i].line, lines[j].line, pt)) {
                    intersectionCounts[i]++;
                    intersectionCounts[j]++;
                }
            }
        }

        vector<LineInfo> filtered;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (intersectionCounts[i] <= maxIntersections) {
                filtered.push_back(lines[i]);
            }
        }
        return filtered;
    }

    LineInfo findBestStopLine(const vector<LineInfo>& lines) {
        if (lines.empty())
            throw runtime_error("Nessuna linea orizzontale valida trovata!");
        return *max_element(lines.begin(), lines.end(),
                            [](const LineInfo& a, const LineInfo& b) { return a.score < b.score; });
    }

    void displayResults(const Mat& originalImg, const Mat& edges,
                        const vector<Vec4i>& allLines,
                        const vector<LineInfo>& horizontalLines,
                        const vector<LineInfo>& filteredLines,
                        const LineInfo& stopLine) {
        Mat imgAllLines = originalImg.clone();
        Mat imgHorizontal = originalImg.clone();
        Mat imgFiltered = originalImg.clone();
        Mat imgStopLine = originalImg.clone();

        for (const auto& l : allLines)
            line(imgAllLines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(200, 200, 0), 2);

        for (const auto& l : horizontalLines)
            line(imgHorizontal, Point(l.line[0], l.line[1]), Point(l.line[2], l.line[3]), Scalar(0, 255, 0), 2);

        for (const auto& l : filteredLines)
            line(imgFiltered, Point(l.line[0], l.line[1]), Point(l.line[2], l.line[3]), Scalar(255, 0, 0), 2);

        drawStopLine(imgStopLine, stopLine);

        namedWindow("Tutte le linee rilevate", WINDOW_NORMAL);
        namedWindow("Linee orizzontali", WINDOW_NORMAL);
        namedWindow("Linee filtrate", WINDOW_NORMAL);
        namedWindow("Linea STOP finale", WINDOW_NORMAL);

        imshow("Tutte le linee rilevate", imgAllLines);
        imshow("Linee orizzontali", imgHorizontal);
        imshow("Linee filtrate", imgFiltered);
        imshow("Linea STOP finale", imgStopLine);
    }

    void drawStopLine(Mat& img, const LineInfo& stopLine) {
        const Vec4i& line = stopLine.line;
        Point p1(line[0], line[1]), p2(line[2], line[3]);
        cv::line(img, p1, p2, Scalar(0, 0, 255), 5);
        putText(img, "STOP LINE", Point(p1.x, p1.y - 15), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    }

    bool getIntersectionPoint(const Vec4i& l1, const Vec4i& l2, Point2f& intersection) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        Point2f r = p2 - p1;
        Point2f s = q2 - q1;
        float rxs = r.x * s.y - r.y * s.x;
        float q_pxr = (q1 - p1).x * r.y - (q1 - p1).y * r.x;

        if (rxs == 0 && q_pxr == 0) return false;
        if (rxs == 0 && q_pxr != 0) return false;

        float t = ((q1 - p1).x * s.y - (q1 - p1).y * s.x) / rxs;
        float u = ((q1 - p1).x * r.y - (q1 - p1).y * r.x) / rxs;

        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            intersection = p1 + t * r;
            return true;
        }
        return false;
    }

    void printDetailedInfo(const vector<Vec4i>& allLines,
                           const vector<LineInfo>& filteredLines,
                           const LineInfo& stopLine) {
        cout << "\n========== ANALISI DETTAGLIATA ==========" << endl;
        cout << "Totale linee rilevate: " << allLines.size() << endl;
        cout << "Linee orizzontali dopo filtro intersezioni: " << filteredLines.size() << endl;

        cout << "\n=== STOP LINE IDENTIFICATA ===" << endl;
        const Vec4i& line = stopLine.line;
        cout << "Coordinati: P1(" << line[0] << "," << line[1] << ") "
             << "P2(" << line[2] << "," << line[3] << ")" << endl;
        cout << "Lunghezza: " << (int)stopLine.length << " pixel" << endl;
        cout << "Altezza media Y: " << stopLine.avgY << " pixel" << endl;
        cout << "Angolo: " << (int)stopLine.angle << "°" << endl;
        cout << "Direzione: " << stopLine.direction() << endl;
        cout << "Score: " << (int)stopLine.score << endl;
        cout << "=========================================" << endl;
    }
};

int main() {
    try {
        string imagePath = "C:/Users/Principale/Desktop/stop-line-1.jpg";
        StopLineDetector detector;
        detector.processImage(imagePath);
        cout << "\nPremi un tasto qualsiasi per chiudere le finestre..." << endl;
        waitKey(0);
        destroyAllWindows();
    } catch (const exception& e) {
        cerr << "Errore: " << e.what() << endl;
        return -1;
    }
    return 0;
}

/* 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    // Carica l'immagine
    Mat image = imread("C:/Users/Principale/Documents/GitHub/CV-final-project/images/vlcsnap-2025-07-15-20h51m43s637.png");
    if (image.empty()) {
        cout << "Errore: impossibile caricare l'immagine" << endl;
        return -1;
    }
    
    Mat gray, blur, edges;
    
    // Preprocessing
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 1.4);
    Canny(blur, edges, 50, 150);
     
        int houghThreshold = 150;
        double houghMinLineLength = 500.0;
        double houghMaxLineGap = 100.0;
        double horizontalTolerance = 15.0;
        double minLineLength = 80.0;
        int maxIntersections = 3;
    
    // Rilevamento delle linee con Hough Transform
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 150, 500, 300);
    
    // Trova il punto di fuga
    vector<Point2f> intersections;
    
    // Calcola tutte le intersezioni tra le linee
    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            // Linea 1
            float x1 = lines[i][0], y1 = lines[i][1];
            float x2 = lines[i][2], y2 = lines[i][3];
            
            // Linea 2
            float x3 = lines[j][0], y3 = lines[j][1];
            float x4 = lines[j][2], y4 = lines[j][3];
            
            // Calcola l'intersezione
            float denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
            
            if (abs(denom) > 0.01) { // Evita divisione per zero
                float px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom;
                float py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom;
                
                // Filtra punti troppo lontani
                if (px > -image.cols && px < 2*image.cols && 
                    py > -image.rows && py < 2*image.rows) {
                    intersections.push_back(Point2f(px, py));
                }
            }
        }
    }
    
    // Trova il punto di fuga (media delle intersezioni)
    Point2f vanishingPoint(0, 0);
    if (!intersections.empty()) {
        for (const auto& point : intersections) {
            vanishingPoint += point;
        }
        vanishingPoint.x /= intersections.size();
        vanishingPoint.y /= intersections.size();
    }
    
    // Visualizza i risultati
    Mat result = image.clone();
    
    // Disegna tutte le linee rilevate
    for (size_t i = 0; i < lines.size(); i++) {
        line(result, Point(lines[i][0], lines[i][1]), 
             Point(lines[i][2], lines[i][3]), Scalar(0, 255, 0), 2);
    }
    
    // Disegna il punto di fuga
    if (!intersections.empty()) {
        circle(result, vanishingPoint, 10, Scalar(0, 0, 255), -1);
        cout << "Punto di fuga: (" << vanishingPoint.x << ", " << vanishingPoint.y << ")" << endl;
        
        // Disegna alcune linee di prospettiva
        for (size_t i = 0; i < min(lines.size(), (size_t)10); i++) {
            line(result, vanishingPoint, Point(lines[i][0], lines[i][1]), Scalar(255, 0, 0), 1);
            line(result, vanishingPoint, Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0), 1);
        }
    } else {
        cout << "Punto di fuga non trovato" << endl;
    }
    
    // Crea finestre ridimensionabili
    namedWindow("Originale", WINDOW_NORMAL);
    namedWindow("Bordi", WINDOW_NORMAL);
    namedWindow("Punto di Fuga", WINDOW_NORMAL);
    
    // Ridimensiona le finestre per adattarle allo schermo
    resizeWindow("Originale", 800, 600);
    resizeWindow("Bordi", 800, 600);
    resizeWindow("Punto di Fuga", 800, 600);
    
    // Mostra i risultati
    imshow("Originale", image);
    imshow("Bordi", edges);
    imshow("Punto di Fuga", result);
    
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main() {
    // Carica immagine in scala di grigi
    cv::Mat img = cv::imread("C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m43s637.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img = cv::imread("C:\\Users\\Principale\\Desktop\\image1.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img = cv::imread("C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-17-00h29m56s656.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Errore: immagine non trovata." << std::endl;
        return -1;
    }

    // Normalizzazione per evidenziare i dettagli
    cv::Mat blurStrong, normalized;
    cv::GaussianBlur(img, blurStrong, cv::Size(31, 31), 0);
    cv::absdiff(img, blurStrong, normalized);

    // Binarizzazione con Otsu
    cv::Mat binary;
//cv::threshold(normalized, binary, 30, 255, cv::THRESH_BINARY);
cv::threshold(normalized, binary, 30, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);



    // Visualizza immagine binaria corretta
    cv::namedWindow("Immagine binaria (threshold)", cv::WINDOW_NORMAL);
    cv::imshow("Immagine binaria (threshold)", normalized);

    // 5. Trova i contorni
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 6. Visualizza i contorni per debug
    cv::Mat img_contours;
    cv::cvtColor(binary, img_contours, cv::COLOR_GRAY2BGR);
    cv::drawContours(img_contours, contours, -1, cv::Scalar(0, 255, 0), 1); // verde
    cv::namedWindow("Contorni trovati", cv::WINDOW_NORMAL);
    cv::imshow("Contorni trovati", img_contours);

    // 5. Crea immagine finale vuota
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC1);
    std::vector<cv::Point2f> detectedPoints;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 20)
            continue;

        if (contour.size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            if (ellipse.size.width > 10 && ellipse.size.height > 5 &&
                ellipse.size.width < 60 && ellipse.size.height < 50) {
                cv::ellipse(output, ellipse, 255, -1);
                detectedPoints.push_back(ellipse.center);
                continue;
            }
        }
    }

    double dist = 50

    for (const auto& point : detectedPoints) 
    {
        
    }



    // 6. Morfologia (pulizia)
    //cv::morphologyEx(output, output, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 10)));

    // 7. Mostra risultato finale
    cv::namedWindow("Segmentazione Finale", cv::WINDOW_NORMAL);
    cv::imshow("Segmentazione Finale", output);

    cv::waitKey(0);
    return 0;
}
*/