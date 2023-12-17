#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class KLTTracker
{
public:
	KLTTracker(const std::string path);
	~KLTTracker();

	void showTrack();
private:
	void computeGradients(const Mat& image, Mat& Ix, Mat& Iy);
	void manualCalcOpticalFlowPyrLK(const Mat& prevImage, const Mat& nextImage,
		const vector<Point2f>& prevPoints, vector<Point2f>& nextPoints,
		vector<uchar>& status, vector<float>& error,
		Size winSize = Size(21, 21), int maxLevel = 3,
		TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.01), int flags = 0,
		double minEigThreshold = 0.0001f);

	cv::VideoCapture cap;
};



