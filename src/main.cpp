/*
 *  main.cpp
 *  Author: Milica Masic
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include "TrafficLightDetector.h"
#include "VehicleDetector.hpp"
#include "StopLineDetector.h"

using namespace cv;
using namespace std;
/*
// select detection parameters based on video
TrafficLightDetector::DetectionParams getDetectionParamsForVideo(
    const std::string& videoName, int cols, int rows) {
    
    TrafficLightDetector::DetectionParams params;

    if (videoName == "video1.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video2.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video3.mp4") {
        params.houghParam1 = 120;
        params.houghParam2 = 25;
        params.minRadius = 8;
        params.maxRadius = 35;
        params.roi = Rect(1670, 170, 90, 210);
    } else if (videoName == "video4.mp4") {
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(80, 550, 100, 250);
    } else if (videoName == "video5.mp4") {
        params.houghParam1 = 30;
        params.houghParam2 = 10;
        params.minRadius = 2;
        params.maxRadius = 10;
        params.roi = Rect(290, 70, 15, 25);
    } else {
        //default
        params.houghParam1 = 100;
        params.houghParam2 = 20;
        params.minRadius = 10;
        params.maxRadius = 40;
        params.roi = Rect(0, 0, cols, rows);
    }

    return params;
}

int main(int argc, char** argv)
{
    //video display
    string videoPath = "../videos/video1.mp4";  
    string videoName = videoPath.substr(videoPath.find_last_of("/\\") + 1);

    //upload the model yolov5 (we had to used an limited versione because it was incompatible with the opncv version of the vlab)
    string modelPath = "../models/yolov5s_clean.onnx"; 
    VehicleDetector detector (modelPath, 0.3f, 0.45f);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) return -1;

    Mat frame;
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = 1000 / fps;  //delay between frames in ms

    namedWindow("Traffic light detection", WINDOW_NORMAL);

    TrafficLightDetector::DetectionParams params = getDetectionParamsForVideo(videoName, frame.cols, frame.rows);
    TrafficLightDetector tlDetector(params);

    while (cap.read(frame)) {

        detector.detect(frame);
        tlDetector.detectAndAnnotate(frame, videoName);
        
        imshow("Traffic light detection", frame);
        if (waitKey(delay) == 27) break;
    }
    destroyAllWindows();

    return 0;
   
}
*/
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// ====================== Funzioni di supporto ======================
// distanza punto-segmento
double distPointSegment(const Point2f& p, const Point2f& a, const Point2f& b) {
    Point2f ab = b - a;
    Point2f ap = p - a;
    double t = (ap.dot(ab)) / (ab.dot(ab) + 1e-9); // evita divisione per zero
    t = std::max(0.0, std::min(1.0, t));
    Point2f proj = a + t * ab;
    return norm(p - proj);
}

// distanza minima tra due segmenti
double segmentDistance(const Point2f& a, const Point2f& b,
                       const Point2f& c, const Point2f& d) {
    double d1 = distPointSegment(a, c, d);
    double d2 = distPointSegment(b, c, d);
    double d3 = distPointSegment(c, a, b);
    double d4 = distPointSegment(d, a, b);
    return std::min({d1, d2, d3, d4});
}

// ====================== MAIN ======================
int main() {
    string filename = "C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m43s637.png";
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Errore: impossibile aprire " << filename << "\n";
        return 1;
    }

    // --- Preprocessing ---
    Mat gray, blurred, edges;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5,5), 1.4);
    Canny(blurred, edges, 50, 150);

    // --- Rilevamento linee ---
    vector<Vec4i> linesP;
    HoughLinesP(edges, linesP, 1, CV_PI/180, 20, 500, 200);

    // --- Filtra solo linee orizzontali ---
    vector<Vec4i> horizLines = linesP; // Copia tutte le linee
    /*for (auto &l : linesP) {
        Point2f p1(l[0], l[1]), p2(l[2], l[3]);
        double angle = atan2(p2.y - p1.y, p2.x - p1.x);
        if (fabs(angle) < CV_PI/36.0 || fabs(fabs(angle) - CV_PI) < CV_PI/36.0) {
            horizLines.push_back(l);
        }
    }*/

    // --- Raggruppamento linee orizzontali ---
    auto predicateline = [](const Vec4i& l1, const Vec4i& l2) {
        Point2f p1(l1[0], l1[1]), p2(l1[2], l1[3]);
        Point2f q1(l2[0], l2[1]), q2(l2[2], l2[3]);

        // Angoli
        double angle1 = atan2(p2.y - p1.y, p2.x - p1.x);
        double angle2 = atan2(q2.y - q1.y, q2.x - q1.x);
        double dAngle = fabs(angle1 - angle2);
        if (dAngle > CV_PI/2) dAngle = CV_PI - dAngle;

        // Distanza minima tra segmenti
        double dist = segmentDistance(p1, p2, q1, q2);

        // Parametri soglia
        double maxDist = 20.0;           // pixel
        double maxAngle = CV_PI / 36.0;  // ~5°

        return (dist < maxDist) && (dAngle < maxAngle);
    };

    vector<int> labels;
    int nLabels = partition(horizLines, labels, predicateline);

    // --- Visualizzazione dei gruppi ---
    Mat debug = img.clone();
    RNG rng(12345); // random per colori
    vector<Scalar> colors(nLabels);
    for (int i = 0; i < nLabels; i++) {
        colors[i] = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
    }

    for (size_t i = 0; i < horizLines.size(); i++) {
        Vec4i l = horizLines[i];
        Scalar col = colors[labels[i]];
        line(debug, Point(l[0], l[1]), Point(l[2], l[3]), col, 3, LINE_AA);
    }

    cout << "Numero di gruppi trovati: " << nLabels << endl;

    // --- Mostra risultato ---
    namedWindow("Contorni per Gruppi", WINDOW_NORMAL);
    imshow("Contorni per Gruppi", debug);
    waitKey(0);

    return 0;
}

/*
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

double getMainOrientation(const vector<Point>& contour) {
    if (contour.size() < 5) return 0.0;
    
    // Calcola i momenti del contorno
    Moments m = moments(contour);
    
    // Calcola l'orientamento principale usando i momenti centrali
    double mu20 = m.mu20;
    double mu02 = m.mu02;
    double mu11 = m.mu11;
    
    // Calcola l'angolo di orientamento
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    return theta;
}

// Funzione per calcolare la differenza angolare normalizzata
double getAngularDifference(double angle1, double angle2) {
    double diff = abs(angle1 - angle2);
    // Normalizza la differenza nell'intervallo [0, π/2]
    while (diff > CV_PI/2) {
        diff = CV_PI - diff;
    }
    return diff;
}


// Funzione di preprocessing
Mat findMask(const Mat& img) {
    if (img.empty()) {
        std::cerr << "Errore: immagine vuota!" << std::endl;
        return Mat();
    }

    // --- Lab + CLAHE ---
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

    // --- Scala di grigi ---
    Mat gray;
    cvtColor(img_eq, gray, COLOR_BGR2GRAY);

    // --- Background estimation ---
    Mat background;
    GaussianBlur(gray, background, Size(51, 51), 0);

    Mat diff;
    absdiff(gray, background, diff);

    Mat enhanced;
    diff.convertTo(enhanced, -1, 2.0, 0);

    // --- Otsu ---
    Mat otsu;
    threshold(enhanced, otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // --- Morfologia ---
    Mat otsu_cleaned;
    Mat kernel_open = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(10,10));

    morphologyEx(otsu, otsu_cleaned, MORPH_OPEN, kernel_open);
    morphologyEx(otsu_cleaned, otsu_cleaned, MORPH_CLOSE, kernel_close);

    return otsu_cleaned;
}

int main() {
    //Mat img = imread("C:\\Users\\Principale\\Desktop\\image1.png", IMREAD_COLOR_BGR);
    //Mat img = imread("C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-17-00h29m38s168.png", IMREAD_COLOR_BGR);
    Mat img = imread("C:\\Users\\Principale\\Documents\\GitHub\\CV-final-project\\images\\vlcsnap-2025-07-15-20h51m43s637.png", IMREAD_COLOR);
    Mat mask = findMask(img);


    auto predicate = [](const vector<Point>& c1, const vector<Point>& c2) {
    // 1. Controllo di similarità di forma (invariante alle trasformazioni)
    double shapeDistance = matchShapes(c1, c2, CONTOURS_MATCH_I1, 0.0);
    
    // Soglia per la similarità di forma
    const double SHAPE_THRESHOLD = 0.3;
    if (shapeDistance > SHAPE_THRESHOLD) {
        return false; // Forme troppo diverse
    }
    
    // 2. Controllo di allineamento (orientamento)
    double orientation1 = getMainOrientation(c1);
    double orientation2 = getMainOrientation(c2);
    double angularDiff = getAngularDifference(orientation1, orientation2);
    
    // Soglia per l'allineamento (15 gradi in radianti)
    const double ALIGNMENT_THRESHOLD = 15.0 * CV_PI / 180.0;
    
    // Ritorna true se i contorni sono simili E allineati
    return (shapeDistance < SHAPE_THRESHOLD) && (angularDiff < ALIGNMENT_THRESHOLD);
    };


    if (!mask.empty()) {
    // Trova contorni
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double minArea = 100.0;    // es. minimo 100 pixel
double maxArea = 5000.0;   // es. massimo 5000 pixel

vector<vector<Point>> filteredContours;
for (const auto& c : contours) {
    double area = contourArea(c);
    if (area >= minArea && area <= maxArea) {
        filteredContours.push_back(c);
    }
}

cout << "Contorni trovati (totali): " << filteredContours.size() << endl;


    vector<int> labels;
    int nLabels = partition(contours, labels, predicate);

cout << "gay: " << endl;



        // Disegna contorni sull’immagine originale
    Mat img_contours = img.clone();
        drawContours(img_contours, contours, -1, Scalar(0, 0, 255), 2);

    // Mostra i risultati
    namedWindow("Mask", WINDOW_NORMAL);
    imshow("Mask", mask);

    namedWindow("Contours", WINDOW_NORMAL);
    imshow("Contours", img_contours);

    waitKey(0);
}
    return 0;
}
*/