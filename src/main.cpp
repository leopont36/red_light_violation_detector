#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;

// ====================== Configurazione parametri ======================
struct DetectionParams {
    // Preprocessing
    double claheClipLimit = 2.0;
    Size claheTileSize = Size(8, 8);
    
    // Canny
    double cannyLowThresholdFactor = 0.66;
    double cannyHighThresholdFactor = 1.33;
    
    // Hough
    int houghThreshold = 50;
    double minLineLengthRatio = 0.20;  // % larghezza immagine
    double maxLineGapRatio = 0.10;
    
    // Filtri geometrici
    double maxHorizontalAngleDeg = 15.0;
    double minHorizontalCoverage = 0.15;
    double roiLowerThirdOnly = 0.33;  // ignora top 33%
    
    // Raggruppamento
    double groupDistanceThreshold = 20.0;
    double groupAngleThresholdDeg = 5.0;
    
    // Scoring
    double weightThickness = 0.20;
    double weightBottomPosition = 0.35;
    double weightCoverage = 0.30;
    double weightLength = 0.15;
};

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
Rect computeROI(const Mat& img, const DetectionParams& params) {
    int topCut = img.rows * params.roiLowerThirdOnly;
    return Rect(0, topCut, img.cols, img.rows - topCut);
}

// Stima automatica soglie Canny usando metodo Otsu
pair<double, double> estimateCannyThresholds(const Mat& blurred, const DetectionParams& params) {
    Scalar mu, sigma;
    meanStdDev(blurred, mu, sigma);
    double median = mu[0];
    double lower = max(0.0, median * params.cannyLowThresholdFactor);
    double upper = min(255.0, median * params.cannyHighThresholdFactor);
    return {lower, upper};
}

// Versione con informazioni di debug
/*pair<double, double> estimateCannyThresholds(const Mat& blurred, 
                                               const DetectionParams& params
                                               ) {
    Mat gray;
    if (blurred.channels() == 3) {
        cvtColor(blurred, gray, COLOR_BGR2GRAY);
    } else {
        gray = blurred;
    }
    
    // Calcola soglia Otsu
    Mat binarized;
    double otsuThreshold = threshold(gray, binarized, 0, 255, 
                                      THRESH_BINARY | THRESH_OTSU);
    
    // Calcola soglie Canny
    double lower = max(0.0, otsuThreshold * params.cannyLowThresholdFactor);
    double upper = min(255.0, otsuThreshold * params.cannyHighThresholdFactor);
    
    // Validazione
    if (upper <= lower) {
        upper = lower * 2.0;
        upper = min(255.0, upper);
    }
    
    
        cout << "Otsu threshold: " << otsuThreshold << endl;
        cout << "Canny lower: " << lower << endl;
        cout << "Canny upper: " << upper << endl;
        cout << "Ratio: " << (upper / lower) << ":1" << endl;
    
    
    return {lower, upper};
}
/*
pair<double, double> estimateCannyThresholds(const Mat& blurred, const DetectionParams& params) {
    Mat gray;
    if (blurred.channels() == 3) {
        cvtColor(blurred, gray, COLOR_BGR2GRAY);
    } else {
        gray = blurred;
    }

    // Calcola gradienti
    Mat gradX, gradY;
    Sobel(gray, gradX, CV_32F, 1, 0, 3);
    Sobel(gray, gradY, CV_32F, 0, 1, 3);

    // Magnitudine (sqrt(gradX² + gradY²))
    Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);  // Più efficiente

    // Appiattisci in un vettore per percentile
    Mat magFlat = magnitude.reshape(1, magnitude.total());
    cv::sort(magFlat, magFlat, SORT_EVERY_COLUMN + SORT_ASCENDING);

    // Percentili corretti (33° e 66° percentile)
    int idxLow  = static_cast<int>(0.33 * magFlat.total());
    int idxHigh = static_cast<int>(0.66 * magFlat.total());

    double lower = magFlat.at<float>(idxLow);
    double upper = magFlat.at<float>(idxHigh);

    // Applica i fattori di scaling
    lower *= params.cannyLowThresholdFactor;
    upper *= params.cannyHighThresholdFactor;

    // Assicurati che upper > lower con rapporto tipico 2:1 o 3:1
    if (upper <= lower) {
        upper = lower * 2.0;
    }

    // Clamp ai limiti validi
    lower = std::clamp(lower, 0.0, 255.0);
    upper = std::clamp(upper, lower + 1.0, 255.0);

    cout << "Estimated Canny thresholds:" << endl;
    cout << "Lower: " << lower << ", Upper: " << upper << endl;
    cout << "Ratio: " << (upper / lower) << ":1" << endl;

    return {lower, upper};
}*/


// Preprocessing adattivo con morfologia
Mat adaptivePreprocess(const Mat& img, Mat& edges, const Rect& roi, const DetectionParams& params) {
    Mat gray, roiGray, enhanced;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    roiGray = gray(roi);
    
    // CLAHE per contrasto locale
    Ptr<CLAHE> clahe = createCLAHE(params.claheClipLimit, params.claheTileSize);
    clahe->apply(roiGray, enhanced);
    
    // Morfologia per enfatizzare linee orizzontali
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 3));
    Mat closed;
    morphologyEx(enhanced, closed, MORPH_CLOSE, kernel);
    
    // Blur e Canny
    Mat blurred;
    GaussianBlur(closed, blurred, Size(5,5), 1.4);

    
    auto [lower, upper] = estimateCannyThresholds(blurred, params);
    Canny(blurred, edges, lower, upper);
    
    
    
    return enhanced;
}

// Calcola metriche di una linea
struct LineMetrics {
    double length;
    double horizontalCoverage;
    double avgY;
    double angle;
    
    LineMetrics(const Vec4i& line, int imgWidth, int imgHeight) {
        Point2f p1(line[0], line[1]), p2(line[2], line[3]);
        length = norm(p2 - p1);
        horizontalCoverage = fabs(line[2] - line[0]) / (double)imgWidth;
        avgY = (line[1] + line[3]) / 2.0;
        angle = atan2(line[3] - line[1], line[2] - line[0]);
    }
};

// Filtraggio avanzato delle linee
vector<Vec4i> filterCandidateLines(const vector<Vec4i>& lines, const Mat& img, 
                                   const Rect& roi, const DetectionParams& params) {
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

// Raggruppa linee simili
struct LineGroup {
    vector<Vec4i> lines;
    int label;
    
    double totalLength;
    double avgY;
    double avgCoverage;
    int thickness;
    double score;
    
    void computeMetrics(int imgWidth, int imgHeight, const DetectionParams& params) {
        thickness = lines.size();
        totalLength = 0.0;
        double ySum = 0.0;
        avgCoverage = 0.0;
        
        for (auto &l : lines) {
            LineMetrics m(l, imgWidth, imgHeight);
            totalLength += m.length;
            ySum += m.avgY;
            avgCoverage += m.horizontalCoverage;
        }
        
        avgY = ySum / thickness;
        avgCoverage /= thickness;
        
        // Scoring normalizzato
        double bottomScore = avgY / imgHeight;
        double lengthScore = totalLength / imgWidth;
        
        score = params.weightThickness * (thickness / 10.0) +
                params.weightBottomPosition * bottomScore +
                params.weightCoverage * avgCoverage +
                params.weightLength * lengthScore;
    }
    
    void print() const {
        cout << "  Gruppo " << label << ": "
             << "lines=" << thickness
             << ", avgY=" << (int)avgY
             << ", coverage=" << avgCoverage
             << ", score=" << score << endl;
    }
};

vector<LineGroup> groupLines(const vector<Vec4i>& lines, const DetectionParams& params,
                             int imgWidth, int imgHeight) {
    auto predicateline = [&params](const Vec4i& l1, const Vec4i& l2) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI/2) dAngle = CV_PI - dAngle;

        double dist = segmentDistance(p1, p2, q1, q2);
        double maxAngle = params.groupAngleThresholdDeg * CV_PI / 180.0;

        return (dist < params.groupDistanceThreshold) && (dAngle < maxAngle);
    };

    vector<int> labels;
    int nLabels = partition(lines, labels, predicateline);
    
    vector<LineGroup> groups;
    for (int i = 0; i < nLabels; i++) {
        LineGroup group;
        group.label = i;
        
        for (size_t j = 0; j < lines.size(); j++) {
            if (labels[j] == i) {
                group.lines.push_back(lines[j]);
            }
        }
        
        if (!group.lines.empty()) {
            group.computeMetrics(imgWidth, imgHeight, params);
            groups.push_back(group);
        }
    }
    
    // Ordina per score decrescente
    sort(groups.begin(), groups.end(), 
         [](const LineGroup& a, const LineGroup& b) { return a.score > b.score; });
    
    return groups;
}

// Validazione semantica del gruppo migliore
bool validateStopLine(const LineGroup& group, int imgWidth) {
    // Controlla che almeno una linea copra una porzione significativa
    double maxCoverage = 0.0;
    for (auto &l : group.lines) {
        double cov = fabs(l[2] - l[0]) / (double)imgWidth;
        maxCoverage = max(maxCoverage, cov);
    }
    
    // Almeno una linea deve coprire 40% della larghezza
    if (maxCoverage < 0.4) {
        cout << "Validazione fallita: copertura massima " << maxCoverage << " < 0.4" << endl;
        return false;
    }
    
    // Controlla consistenza verticale (linee non troppo sparse)
    vector<double> yPositions;
    for (auto &l : group.lines) {
        yPositions.push_back((l[1] + l[3]) / 2.0);
    }
    
    auto [minY, maxY] = minmax_element(yPositions.begin(), yPositions.end());
    double verticalSpread = *maxY - *minY;
    
    if (verticalSpread > 50) {  // max 50 pixel di spread verticale
        cout << "Validazione fallita: spread verticale " << verticalSpread << " > 50px" << endl;
        return false;
    }
    
    return true;
}

// Visualizzazione avanzata
Mat visualizeResults(const Mat& img, const vector<LineGroup>& groups, int bestGroupIdx) {
    Mat debug = img.clone();
    
    // Disegna tutti i gruppi
    RNG rng(12345);
    for (const auto& group : groups) {
        Scalar color;
        int thickness;
        
        if (group.label == bestGroupIdx) {
            color = Scalar(0, 0, 255);  // Rosso per il migliore
            thickness = 5;
        } else {
            color = Scalar(rng.uniform(100,255), rng.uniform(100,255), rng.uniform(100,255));
            thickness = 2;
        }
        
        for (const auto& l : group.lines) {
            line(debug, Point(l[0], l[1]), Point(l[2], l[3]), color, thickness, LINE_AA);
        }
        
        // Label del gruppo
        if (!group.lines.empty()) {
            Point2f center(0, 0);
            for (const auto& l : group.lines) {
                center += Point2f((l[0] + l[2])/2.0, (l[1] + l[3])/2.0);
            }
            center *= 1.0 / group.lines.size();
            
            string label = "G" + to_string(group.label) + 
                          " (" + to_string((int)(group.score * 100)) + ")";
            putText(debug, label, center, FONT_HERSHEY_SIMPLEX, 0.8, 
                   color, 2, LINE_AA);
        }
    }
    
    return debug;
}

// ====================== MAIN ======================
int main() {
    // Configurazione
    DetectionParams params;
    //string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m43s637.png";  
    string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-17-00h29m38s168.png";
    //string filename = "C:\\Users\\Principale\\Desktop\\image1.png";

    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Errore: impossibile aprire " << filename << "\n";
        return 1;
    }

    cout << "=== STOP LINE DETECTOR ===" << endl;
    cout << "Dimensioni immagine: " << img.cols << "x" << img.rows << endl;

    // --- 1. ROI intelligente ---
    Rect roi = computeROI(img, params);
    cout << "\nROI: " << roi << endl;

    // --- 2. Preprocessing adattivo ---
    Mat edges;
    Mat enhanced = adaptivePreprocess(img, edges, roi, params);

    // --- 3. Rilevamento linee con Hough ---
    int minLineLength = img.cols * params.minLineLengthRatio;
    int maxLineGap = img.cols * params.maxLineGapRatio;
    
    cout << "\nParametri Hough:" << endl;
    cout << "  minLineLength: " << minLineLength << endl;
    cout << "  maxLineGap: " << maxLineGap << endl;
    
    vector<Vec4i> linesP;
    HoughLinesP(edges, linesP, 1, CV_PI/180, params.houghThreshold, 
                minLineLength, maxLineGap);
    
    cout << "Linee rilevate (raw): " << linesP.size() << endl;

    if (linesP.empty()) {
        cout << "Nessuna linea rilevata!" << endl;
        imshow("Edges", edges);
        waitKey(0);
        return 0;
    }

    // --- 4. Filtraggio candidati ---
    vector<Vec4i> candidates = filterCandidateLines(linesP, img, roi, params);

    if (candidates.empty()) {
        cout << "Nessuna linea candidata dopo filtri!" << endl;
        imshow("Edges", edges);
        waitKey(0);
        return 0;
    }

    // --- 5. Raggruppamento ---
    cout << "\nRaggruppamento linee..." << endl;
    vector<LineGroup> groups = groupLines(candidates, params, img.cols, img.rows);
    
    cout << "\nGruppi trovati: " << groups.size() << endl;
    for (const auto& g : groups) {
        g.print();
    }

    // --- 6. Selezione gruppo migliore ---
    int bestGroupIdx = -1;
    if (!groups.empty()) {
        bestGroupIdx = groups[0].label;  // Già ordinato per score
        
        // --- 7. Validazione semantica ---
        cout << "\nValidazione gruppo migliore..." << endl;
        bool isValid = validateStopLine(groups[0], img.cols);
        
        if (!isValid) {
            cout << "ATTENZIONE: Il gruppo migliore non supera la validazione!" << endl;
            // Prova con il secondo migliore
            if (groups.size() > 1) {
                cout << "Provo con il secondo gruppo..." << endl;
                if (validateStopLine(groups[1], img.cols)) {
                    bestGroupIdx = groups[1].label;
                    cout << "Secondo gruppo valido!" << endl;
                } else {
                    bestGroupIdx = -1;
                }
            }
        } else {
            cout << "Validazione OK!" << endl;
        }
    }

    // --- 8. Visualizzazione ---
    Mat debug = visualizeResults(img, groups, bestGroupIdx);
    
    // ROI overlay
    rectangle(debug, roi, Scalar(255, 255, 0), 2);
    putText(debug, "ROI", Point(roi.x + 10, roi.y + 30), 
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2);

    // Info finale
    if (bestGroupIdx >= 0) {
        const auto& best = *find_if(groups.begin(), groups.end(),
                                    [bestGroupIdx](const LineGroup& g) { 
                                        return g.label == bestGroupIdx; 
                                    });
        string info = "STOP LINE: Score=" + to_string((int)(best.score * 100)) +
                     " Lines=" + to_string(best.thickness);
        putText(debug, info, Point(20, 40), FONT_HERSHEY_SIMPLEX, 
                1.0, Scalar(0, 255, 0), 2, LINE_AA);
    } else {
        putText(debug, "NESSUNA STOP LINE RILEVATA", Point(20, 40), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Mostra risultati
    namedWindow("Stop Line Detection", WINDOW_NORMAL);
    imshow("Stop Line Detection", debug);
    
    namedWindow("Preprocessing (CLAHE)", WINDOW_NORMAL);
    imshow("Preprocessing (CLAHE)", enhanced);
    
    // Edges in ROI
    Mat edgesFull = Mat::zeros(img.size(), CV_8UC1);
    edges.copyTo(edgesFull(roi));
    namedWindow("Edge Detection (ROI)", WINDOW_NORMAL);
    imshow("Edge Detection (ROI)", edgesFull);
    
    waitKey(0);
    return 0;
}
