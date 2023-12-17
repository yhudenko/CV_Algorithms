#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct BrightnessCounter
{
	int brighter;
	int darker;
};

class CornerDetectorFAST
{
public:
	CornerDetectorFAST(const std::string path);
	~CornerDetectorFAST();
	cv::Mat detectFAST(int threshold);

	bool isSaveImage = true;
	const int radius = 3;
	const cv::Point offsetPoints[16] = {
		{0, -3},
		{1, -3},
		{2, -2},
		{3, -1},
		{3, 0},
		{3, 1},
		{2, 2},
		{1, 3},
		{0, 3},
		{-1, 3},
		{-2, 2},
		{-3, 1},
		{-3, 0},
		{-3, -1},
		{-2, -2},
		{-1, -3}
	};

private:
	bool fastCheck(int& x, int& y, int& threshold);
	bool fullCheck(int& x, int& y, int& threshold);
	int pointCheck(int x, int y, int& threshold, uchar& centralIntensity, BrightnessCounter& counter);

	cv::Mat inputImage;
	cv::Mat outputImage;
	std::vector<cv::KeyPoint> keypoints;
};

