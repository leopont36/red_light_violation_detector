cmake_minimum_required(VERSION 3.10)
project(CV-FINAL-PROJECT)

set(CMAKE_CXX_STANDARD 17)

# find all the file .cpp in the src fold, and add them
file(GLOB SOURCES "src/*.cpp")
add_executable(CV-FINAL-PROJECT ${SOURCES})

