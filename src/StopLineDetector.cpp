<<<<<<< HEAD
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*
class StopLineDetector 
{
    public:
        static void detectStopLines(const Mat& frame) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // Apply Canny edge detection
            Mat edges;
            Canny(gray, edges, 50, 150);

            // Detect lines using Hough Transform
            vector<Vec2f> lines;
            HoughLines(edges, lines, 1, CV_PI / 180, 100);

            // Draw detected lines on the original frame
            for (size_t i = 0; i < lines.size(); i++) {
                float rho = lines[i][0];
                float theta = lines[i][1];
                Point pt1, pt2;

                double a = cos(theta);
                double b = sin(theta);
                double x0 = a * rho;
                double y0 = b * rho;

                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));

                line(frame, pt1, pt2, Scalar(0, 255, 0), 2);
            }

            // Display the result
            imshow("Stop Line Detection", frame);
            waitKey(0);
        }
    }
}*/
=======
/*
 *  StopLineDetector.cpp
 *  Author: Leonardo Pontello
 */
>>>>>>> 2126c4b47b72d505faed8e1b13752fe481ff5d91
