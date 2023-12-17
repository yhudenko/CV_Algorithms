#include "CornerDetectorFAST.h"
#include <iostream>
#include <cmath>

CornerDetectorFAST::CornerDetectorFAST(const std::string path)
{
    inputImage = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) std::cerr << "Error: Could not read the image." << std::endl;
}

CornerDetectorFAST::~CornerDetectorFAST()
{
}

cv::Mat CornerDetectorFAST::detectFAST(int threshold)
{
    for (int y = radius; y < inputImage.rows - radius; ++y) {
        for (int x = radius; x < inputImage.cols - radius; ++x) {
            uchar centralIntensity = inputImage.at<uchar>(y, x);
            BrightnessCounter counter = { 0, 0 };
            bool isCorner = false;

            ////Fast check
            //pointCheck(x + offsetPoints[0].x, y + offsetPoints[0].y, threshold, centralIntensity, counter);
            //pointCheck(x + offsetPoints[4].x, y + offsetPoints[4].y, threshold, centralIntensity, counter);
            //pointCheck(x + offsetPoints[8].x, y + offsetPoints[8].y, threshold, centralIntensity, counter);
            //pointCheck(x + offsetPoints[12].x, y + offsetPoints[12].y, threshold, centralIntensity, counter);

            //if (counter.brighter < 3 && counter.darker < 3) continue;

            int score = 0;

            //Full check
            for (int i = 0; i < 16; ++i)
            {
                score += pointCheck(x + offsetPoints[i].x, y + offsetPoints[i].y, threshold, centralIntensity, counter);
            }

            if (counter.brighter >= 9 || counter.darker >= 9) isCorner = true;

            if (isCorner) {
                keypoints.emplace_back(x, y, 7, -1, score, -1);
            }
        }
    }
    cv::KeyPointsFilter::retainBest(keypoints, 200);
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

    for (auto point : offsetPoints)
    {
        pointCheck(x + point.x, y + point.y, threshold, centralIntensity, counter);
    }

    if (counter.brighter >= 9 || counter.darker >= 9) return true;
    else return false;
}

int CornerDetectorFAST::pointCheck(int x, int y, int& threshold, uchar& centralIntensity, BrightnessCounter& counter)
{
    const uchar offsetIntensity = inputImage.at<uchar>(y, x);
    if (offsetIntensity >= centralIntensity + threshold) {
        counter.brighter++;
        return offsetIntensity - centralIntensity;
    }
    else if (offsetIntensity <= centralIntensity - threshold) {
        counter.darker++;
        return centralIntensity - offsetIntensity;
    }
}