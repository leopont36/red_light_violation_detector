cmake_minimum_required(VERSION 3.5)
project(CV_FINAL_PROJECT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(detector_lib STATIC
    src/trafficlight_detector.cpp
    src/vehicle_detector.cpp
    src/stopline_detector.cpp
    src/violation_detector.cpp
    src/metrics.cpp
)

target_include_directories(detector_lib PUBLIC
    include/
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(detector_lib PUBLIC
    ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE detector_lib)