#include <opencv2/opencv.hpp>
#include <iostream>

// g++ check.cpp -o check -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
}