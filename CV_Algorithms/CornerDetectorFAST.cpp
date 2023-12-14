#include "CornerDetectorFAST.h"
#include <iostream>

CornerDetectorFAST::CornerDetectorFAST(const std::string path)
{
    inputImage = cv::imread(path, cv::IMREAD_GRAYSCALE);
}

CornerDetectorFAST::~CornerDetectorFAST()
{
}

cv::Mat CornerDetectorFAST::detectFAST(int threshold)
{
    for (int y = radius; y < inputImage.rows - radius; ++y) {
        for (int x = radius; x < inputImage.cols - radius; ++x) {
            if (fastCheck(x, y, threshold)) {
                keypoints.emplace_back(x, y, 7, -1, 0, -1);
            }
        }
    }
    cv::drawKeypoints(inputImage, keypoints, outputImage, cv::Scalar(0, 255, 0));
    return outputImage;
}

bool CornerDetectorFAST::fastCheck(int& x, int& y, int& threshold)
{
    uchar centralIntensity = inputImage.at<uchar>(y, x);
    BrightnessCounter counter = { 0, 0 };

    pointCheck(x + offsetPoints[0].x, y + offsetPoints[0].y, threshold, centralIntensity, counter);
    pointCheck(x + offsetPoints[4].x, y + offsetPoints[4].y, threshold, centralIntensity, counter);
    pointCheck(x + offsetPoints[8].x, y + offsetPoints[8].y, threshold, centralIntensity, counter);
    pointCheck(x + offsetPoints[12].x, y + offsetPoints[12].y, threshold, centralIntensity, counter);

    if (counter.brighter < 3 && counter.darker < 3) return false;

    return fullCheck(x, y, threshold);
}

bool CornerDetectorFAST::fullCheck(int& x, int& y, int& threshold)
{
    uchar centralIntensity = inputImage.at<uchar>(y, x);
    BrightnessCounter counter = { 0, 0 };

    auto offsetPointCheck = [&centralIntensity, &counter](int index) {
        
    };
    for (auto point : offsetPoints)
    {
        pointCheck(x + point.x, y + point.y, threshold, centralIntensity, counter);
    }

    if (counter.brighter >= 9 || counter.darker >= 9) return true;
    else return false;
}

void CornerDetectorFAST::pointCheck(int x, int y, int& threshold, uchar& centralIntensity, BrightnessCounter& counter)
{
    const uchar offsetIntensity = inputImage.at<uchar>(y, x);
    if (offsetIntensity >= centralIntensity + threshold) {
        counter.brighter++;
    }
    else if (offsetIntensity <= centralIntensity - threshold) {
        counter.darker++;
    }
}