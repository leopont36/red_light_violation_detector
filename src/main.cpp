#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;
/*
// ====================== Parametri globali ======================
// Preprocessing
const double CLAHE_CLIP_LIMIT = 2.0;
const Size CLAHE_TILE_SIZE = Size(8, 8);

// Canny
const double CANNY_LOW_THRESHOLD_FACTOR = 0.66;
const double CANNY_HIGH_THRESHOLD_FACTOR = 1.33;

// Hough
const int HOUGH_THRESHOLD = 50;
const double MIN_LINE_LENGTH_RATIO = 0.20;
const double MAX_LINE_GAP_RATIO = 0.10;

// Filtri geometrici
const double MAX_HORIZONTAL_ANGLE_DEG = 15.0;
const double MIN_HORIZONTAL_COVERAGE = 0.15;
const double ROI_LOWER_THIRD_ONLY = 0.33;

// Raggruppamento
const double GROUP_DISTANCE_THRESHOLD = 20.0;
const double GROUP_ANGLE_THRESHOLD_DEG = 5.0;

// Scoring
const double WEIGHT_THICKNESS = 0.20;
const double WEIGHT_BOTTOM_POSITION = 0.35;
const double WEIGHT_COVERAGE = 0.30;
const double WEIGHT_LENGTH = 0.15;

// ====================== Funzioni di supporto ======================
double distPointSegment(const Point2f& p, const Point2f& a, const Point2f& b) {
    Point2f ab = b - a;
    Point2f ap = p - a;
    double t = (ap.dot(ab)) / (ab.dot(ab) + 1e-9);
    t = std::max(0.0, std::min(1.0, t));
    Point2f proj = a + t * ab;
    return norm(p - proj);
}

double segmentDistance(const Point2f& a, const Point2f& b,
                       const Point2f& c, const Point2f& d) {
    double d1 = distPointSegment(a, c, d);
    double d2 = distPointSegment(b, c, d);
    double d3 = distPointSegment(c, a, b);
    double d4 = distPointSegment(d, a, b);
    return std::min({d1, d2, d3, d4});
}

// ROI intelligente basata su prospettiva
Rect computeROI(const Mat& img) {
    int topCut = img.rows * ROI_LOWER_THIRD_ONLY;
    return Rect(0, topCut, img.cols, img.rows - topCut);
}

// Stima automatica soglie Canny
pair<double, double> estimateCannyThresholds(const Mat& blurred) {
    Scalar mu, sigma;
    meanStdDev(blurred, mu, sigma);
    double median = mu[0];
    double lower = max(0.0, median * CANNY_LOW_THRESHOLD_FACTOR);
    double upper = min(255.0, median * CANNY_HIGH_THRESHOLD_FACTOR);
    return {lower, upper};
}

// Preprocessing adattivo con morfologia
Mat adaptivePreprocess(const Mat& img, Mat& edges, const Rect& roi) {
    Mat gray, roiGray, enhanced;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    roiGray = gray(roi);
    
    // CLAHE per contrasto locale
    Ptr<CLAHE> clahe = createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE);
    clahe->apply(roiGray, enhanced);
    
    // Morfologia per enfatizzare linee orizzontali
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 3));
    Mat closed;
    morphologyEx(enhanced, closed, MORPH_CLOSE, kernel);
    
    // Blur e Canny
    Mat blurred;
    GaussianBlur(closed, blurred, Size(5,5), 1.4);
    
    auto [lower, upper] = estimateCannyThresholds(blurred);
    Canny(blurred, edges, lower, upper);
    
    return enhanced;
}

// Calcola metriche di una linea
// Ritorna: {length, horizontalCoverage, avgY, angle}
tuple<double, double, double, double> computeLineMetrics(const Vec4i& line, int imgWidth, int imgHeight) {
    Point2f p1(line[0], line[1]), p2(line[2], line[3]);
    double length = norm(p2 - p1);
    double horizontalCoverage = fabs(line[2] - line[0]) / (double)imgWidth;
    double avgY = (line[1] + line[3]) / 2.0;
    double angle = atan2(line[3] - line[1], line[2] - line[0]);
    
    return {length, horizontalCoverage, avgY, angle};
}

// Filtraggio avanzato delle linee
vector<Vec4i> filterCandidateLines(const vector<Vec4i>& lines, const Mat& img, const Rect& roi) {
    vector<Vec4i> filtered;
    double maxAngle = MAX_HORIZONTAL_ANGLE_DEG * CV_PI / 180.0;
    
    for (auto &line : lines) {
        // Trasla coordinate rispetto a ROI
        Vec4i globalLine = line;
        globalLine[1] += roi.y;
        globalLine[3] += roi.y;
        
        auto [length, horizontalCoverage, avgY, angle] = computeLineMetrics(globalLine, img.cols, img.rows);
        
        // Filtri multipli
        bool isHorizontal = (fabs(angle) < maxAngle || 
                            fabs(fabs(angle) - CV_PI) < maxAngle);
        bool hasGoodCoverage = horizontalCoverage > MIN_HORIZONTAL_COVERAGE;
        bool isInLowerPart = avgY > img.rows * ROI_LOWER_THIRD_ONLY;
        
        if (isHorizontal) {
            filtered.push_back(globalLine);
        }
    }
    
    cout << "Linee filtrate: " << filtered.size() << " / " << lines.size() << endl;
    return filtered;
}

// Raggruppa linee simili
// Ritorna: {labels, nLabels}
pair<vector<int>, int> partitionLines(const vector<Vec4i>& lines) {
    auto predicateline = [](const Vec4i& l1, const Vec4i& l2) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI/2) dAngle = CV_PI - dAngle;

        double dist = segmentDistance(p1, p2, q1, q2);
        double maxAngle = GROUP_ANGLE_THRESHOLD_DEG * CV_PI / 180.0;

        return (dist < GROUP_DISTANCE_THRESHOLD) && (dAngle < maxAngle);
    };

    vector<int> labels;
    int nLabels = partition(lines, labels, predicateline);
    
    return {labels, nLabels};
}

// Calcola metriche per un gruppo
// Ritorna: {thickness, avgY, avgCoverage, totalLength, score}
tuple<int, double, double, double, double> computeGroupMetrics(
    const vector<Vec4i>& groupLines, int imgWidth, int imgHeight) {
    
    int thickness = groupLines.size();
    double totalLength = 0.0;
    double ySum = 0.0;
    double avgCoverage = 0.0;
    
    for (auto &l : groupLines) {
        auto [length, coverage, avgY, angle] = computeLineMetrics(l, imgWidth, imgHeight);
        totalLength += length;
        ySum += avgY;
        avgCoverage += coverage;
    }
    
    double avgY = ySum / thickness;
    avgCoverage /= thickness;
    
    // Scoring normalizzato
    double bottomScore = avgY / imgHeight;
    double lengthScore = totalLength / imgWidth;
    
    double score = WEIGHT_THICKNESS * (thickness / 10.0) +
                   WEIGHT_BOTTOM_POSITION * bottomScore +
                   WEIGHT_COVERAGE * avgCoverage +
                   WEIGHT_LENGTH * lengthScore;
    
    return {thickness, avgY, avgCoverage, totalLength, score};
}

// Crea gruppi di linee
// Ritorna vettore di tuple: {label, lines, thickness, avgY, avgCoverage, totalLength, score}
vector<tuple<int, vector<Vec4i>, int, double, double, double, double>> 
groupLines(const vector<Vec4i>& lines, int imgWidth, int imgHeight) {
    
    auto [labels, nLabels] = partitionLines(lines);
    
    vector<tuple<int, vector<Vec4i>, int, double, double, double, double>> groups;
    
    for (int i = 0; i < nLabels; i++) {
        vector<Vec4i> groupLines;
        
        for (size_t j = 0; j < lines.size(); j++) {
            if (labels[j] == i) {
                groupLines.push_back(lines[j]);
            }
        }
        
        if (!groupLines.empty()) {
            auto [thickness, avgY, avgCoverage, totalLength, score] = 
                computeGroupMetrics(groupLines, imgWidth, imgHeight);
            
            groups.push_back({i, groupLines, thickness, avgY, avgCoverage, totalLength, score});
        }
    }
    
    // Ordina per score decrescente
    sort(groups.begin(), groups.end(), 
         [](const auto& a, const auto& b) { 
             return get<6>(a) > get<6>(b); 
         });
    
    return groups;
}

// Validazione semantica del gruppo migliore (solo per feedback, non blocca il risultato)
bool validateStopLine(const vector<Vec4i>& groupLines, int imgWidth) {
    // Controlla che almeno una linea copra una porzione significativa
    double maxCoverage = 0.0;
    for (auto &l : groupLines) {
        double cov = fabs(l[2] - l[0]) / (double)imgWidth;
        maxCoverage = max(maxCoverage, cov);
    }
    
    if (maxCoverage < 0.4) {
        cout << "Validazione fallita: copertura massima " << maxCoverage << " < 0.4" << endl;
        return false;
    }
    
    // Controlla consistenza verticale
    vector<double> yPositions;
    for (auto &l : groupLines) {
        yPositions.push_back((l[1] + l[3]) / 2.0);
    }
    
    auto [minY, maxY] = minmax_element(yPositions.begin(), yPositions.end());
    double verticalSpread = *maxY - *minY;
    
    if (verticalSpread > 50) {
        cout << "Validazione fallita: spread verticale " << verticalSpread << " > 50px" << endl;
        return false;
    }
    
    return true;
}

// ====================== NUOVA FUNZIONE: Calcola linea rappresentativa ======================
Vec4i computeRepresentativeLine(const vector<Vec4i>& groupLines) {
    if (groupLines.empty()) {
        return Vec4i(0, 0, 0, 0);
    }
    
    // Raccoglie tutti i punti del gruppo
    vector<Point2f> points;
    int minX = INT_MAX, maxX = INT_MIN;
    
    for (const auto& line : groupLines) {
        points.push_back(Point2f(line[0], line[1]));
        points.push_back(Point2f(line[2], line[3]));
        minX = min(minX, min(line[0], line[2]));
        maxX = max(maxX, max(line[0], line[2]));
    }
    
    // Fit di una linea con least squares (y = mx + q)
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
    
    Vec4i representativeLine(minX, y1, maxX, y2);
    
    return representativeLine;
}

// Visualizzazione avanzata con linea rappresentativa
Mat visualizeResults(const Mat& img, 
                     const vector<tuple<int, vector<Vec4i>, int, double, double, double, double>>& groups,
                     int bestGroupIdx,
                     const Vec4i& representativeLine) {
    Mat debug = img.clone();
    
    RNG rng(12345);
    for (const auto& group : groups) {
        int label = get<0>(group);
        const auto& lines = get<1>(group);
        double score = get<6>(group);
        
        Scalar color;
        int thickness;
        
        if (label == bestGroupIdx) {
            color = Scalar(0, 0, 255);
            thickness = 5;
        } else {
            color = Scalar(rng.uniform(100,255), rng.uniform(100,255), rng.uniform(100,255));
            thickness = 2;
        }
        
        for (const auto& l : lines) {
            line(debug, Point(l[0], l[1]), Point(l[2], l[3]), color, thickness, LINE_AA);
        }
        
        if (!lines.empty()) {
            Point2f center(0, 0);
            for (const auto& l : lines) {
                center += Point2f((l[0] + l[2])/2.0, (l[1] + l[3])/2.0);
            }
            center *= 1.0 / lines.size();
            
            string labelStr = "G" + to_string(label) + 
                             " (" + to_string((int)(score * 100)) + ")";
            putText(debug, labelStr, center, FONT_HERSHEY_SIMPLEX, 0.8, 
                   color, 2, LINE_AA);
        }
    }
    
    // Disegna la linea rappresentativa in verde brillante
    if (representativeLine[0] != 0 || representativeLine[2] != 0) {
        line(debug, Point(representativeLine[0], representativeLine[1]), 
             Point(representativeLine[2], representativeLine[3]), 
             Scalar(0, 255, 0), 6, LINE_AA);
        
        // Aggiungi label
        Point labelPos(representativeLine[0] + 10, representativeLine[1] - 10);
        putText(debug, "STOP LINE", labelPos, FONT_HERSHEY_SIMPLEX, 
                1.2, Scalar(0, 255, 0), 3, LINE_AA);
    }
    
    return debug;
}

// ====================== MAIN ======================
int main() {
    //string filename = "C:\\Users\\Principale\\Desktop\\image1.png";
    string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m19s421.png";

    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Errore: impossibile aprire " << filename << "\n";
        return 1;
    }

    cout << "=== STOP LINE DETECTOR ===" << endl;
    cout << "Dimensioni immagine: " << img.cols << "x" << img.rows << endl;

    // 1. ROI intelligente
    Rect roi = computeROI(img);
    cout << "\nROI: " << roi << endl;

    // 2. Preprocessing adattivo
    Mat edges;
    Mat enhanced = adaptivePreprocess(img, edges, roi);

    // 3. Rilevamento linee con Hough
    int minLineLength = img.cols * MIN_LINE_LENGTH_RATIO;
    int maxLineGap = img.cols * MAX_LINE_GAP_RATIO;
    
    cout << "\nParametri Hough:" << endl;
    cout << "  minLineLength: " << minLineLength << endl;
    cout << "  maxLineGap: " << maxLineGap << endl;
    
    vector<Vec4i> linesP;
    HoughLinesP(edges, linesP, 1, CV_PI/180, HOUGH_THRESHOLD, 
                minLineLength, maxLineGap);
    
    cout << "Linee rilevate (raw): " << linesP.size() << endl;

    if (linesP.empty()) {
        cout << "Nessuna linea rilevata!" << endl;
        imshow("Edges", edges);
        waitKey(0);
        return 0;
    }

    // 4. Filtraggio candidati
    vector<Vec4i> candidates = filterCandidateLines(linesP, img, roi);

    if (candidates.empty()) {
        cout << "Nessuna linea candidata dopo filtri!" << endl;
        imshow("Edges", edges);
        waitKey(0);
        return 0;
    }

    // 5. Raggruppamento
    cout << "\nRaggruppamento linee..." << endl;
    auto groups = groupLines(candidates, img.cols, img.rows);
    
    cout << "\nGruppi trovati: " << groups.size() << endl;
    for (const auto& g : groups) {
        int label = get<0>(g);
        int thickness = get<2>(g);
        double avgY = get<3>(g);
        double avgCoverage = get<4>(g);
        double score = get<6>(g);
        
        cout << "  Gruppo " << label << ": "
             << "lines=" << thickness
             << ", avgY=" << (int)avgY
             << ", coverage=" << avgCoverage
             << ", score=" << score << endl;
    }

    // 6. Selezione gruppo migliore (SEMPRE il primo, anche senza validazione)
    int bestGroupIdx = -1;
    Vec4i representativeLine(0, 0, 0, 0);
    
    if (!groups.empty()) {
        bestGroupIdx = get<0>(groups[0]);
        const auto& bestLines = get<1>(groups[0]);
        
        // 7. Validazione semantica (solo per feedback)
        cout << "\nValidazione gruppo migliore..." << endl;
        bool isValid = validateStopLine(bestLines, img.cols);
        
        if (isValid) {
            cout << "Validazione OK!" << endl;
        } else {
            cout << "ATTENZIONE: Il gruppo migliore non supera la validazione, ma viene comunque utilizzato." << endl;
        }
        
        // 8. Calcola linea rappresentativa
        representativeLine = computeRepresentativeLine(bestLines);
        
        cout << "\nLinea rappresentativa: " << endl;
        cout << "  P1: (" << representativeLine[0] << ", " << representativeLine[1] << ")" << endl;
        cout << "  P2: (" << representativeLine[2] << ", " << representativeLine[3] << ")" << endl;
    }

    // 9. Visualizzazione
    Mat debug = visualizeResults(img, groups, bestGroupIdx, representativeLine);
    
    rectangle(debug, roi, Scalar(255, 255, 0), 2);
    putText(debug, "ROI", Point(roi.x + 10, roi.y + 30), 
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2);

    if (bestGroupIdx >= 0) {
        auto it = find_if(groups.begin(), groups.end(),
                         [bestGroupIdx](const auto& g) { 
                             return get<0>(g) == bestGroupIdx; 
                         });
        if (it != groups.end()) {
            int thickness = get<2>(*it);
            double score = get<6>(*it);
            string info = "BEST GROUP: Score=" + to_string((int)(score * 100)) +
                         " Lines=" + to_string(thickness);
            putText(debug, info, Point(20, 40), FONT_HERSHEY_SIMPLEX, 
                    1.0, Scalar(0, 255, 0), 2, LINE_AA);
        }
    } else {
        putText(debug, "NESSUNA LINEA RILEVATA", Point(20, 40), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
    }

    namedWindow("Stop Line Detection", WINDOW_NORMAL);
    imshow("Stop Line Detection", debug);
    
    namedWindow("Preprocessing (CLAHE)", WINDOW_NORMAL);
    imshow("Preprocessing (CLAHE)", enhanced);
    
    Mat edgesFull = Mat::zeros(img.size(), CV_8UC1);
    edges.copyTo(edgesFull(roi));
    namedWindow("Edge Detection (ROI)", WINDOW_NORMAL);
    imshow("Edge Detection (ROI)", edgesFull);
    
    waitKey(0);
    return 0;
}*/


#include "StopLineDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    //string filename = "C:\\Users\\Principale\\Desktop\\image1.png";
    //string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m19s421.png";
    string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-17-00h29m56s656.png";
    // Load image
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    // Detect stop line
    StopLineDetector detector;
    Vec4i stopLine = detector.detectStopLine(img);

    // Draw stop line
    line(img, Point(stopLine[0], stopLine[1]), 
              Point(stopLine[2], stopLine[3]), 
              Scalar(0, 255, 0), 3);

    // Show result
    namedWindow("Stop Line Detection", WINDOW_NORMAL);
    imshow("Stop Line Detection", img);
    waitKey(0);

    return 0;
}