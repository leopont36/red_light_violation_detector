# Red Light Violation Detector — Computer Vision

Automatic traffic light violation detection system developed as part of the
Computer Vision course. The system identifies vehicles crossing an intersection
during a red light using a modular pipeline combining traditional computer
vision techniques and deep learning.

## Features
- Traffic light detection and state recognition using Hough Transform and HSV
  color analysis with voting mechanism
- Stop line detection using Canny edge detection, Probabilistic Hough Transform
  and least squares regression
- Vehicle detection using YOLOv5
- Violation detection by combining all modules outputs
- Performance evaluation with accuracy, mIoU and mAP metrics

## Technologies
- C++17
- OpenCV
- YOLOv5
- CMake

## Results
| Metric | Score |
|---|---|
| Traffic Light Accuracy | 0.30 |
| Stopline mean IoU | 0.143 |
| Vehicle mAP | 0.528 |
| Violation Detection Accuracy | 0.70 |

## Report
The full project report is available [here](docs/report.pdf).

## Demo
https://github.com/user-attachments/assets/172f3790-3d11-48d9-bff2-035af8e43ea8

## Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Run
```bash
./CV_FINAL_PROJECT
```

## Contributors
Developed in collaboration with
[@masicm](https://github.com/masicm) and
[@angelicazonta](https://github.com/angelicazonta)
as part of a university group assignment.
