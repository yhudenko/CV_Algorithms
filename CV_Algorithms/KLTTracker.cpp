#include "KLTTracker.h"
#include <iostream>

using namespace cv;
using namespace std;

KLTTracker::KLTTracker(const std::string path)
{
	cap = cv::VideoCapture(path);
	if (!cap.isOpened()) std::cerr << "Error opening video file." << std::endl;
}

KLTTracker::~KLTTracker()
{
	cap.release();
}

void KLTTracker::showTrack()
{
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return;
	}

	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	const int MAX_COUNT = 500;

	namedWindow("KTL", 1);

	bool addRemovePt = false;
	Mat frame, image;
	Mat gray[2];
	Point2f point;
	vector<Point2f> points[2];
	points[0].push_back(Point2f(300, 50));

	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray[0], COLOR_BGR2GRAY);

		if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (gray[1].empty())
				gray[0].copyTo(gray[1]);
			manualCalcOpticalFlowPyrLK(gray[1], gray[0], points[0], points[1], status, err, cv::Size(15, 15), 2, termcrit);
			//calcOpticalFlowPyrLK(gray[1], gray[0], points[0], points[1], status, err, cv::Size(15, 15), 2, termcrit);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray[0], tmp, winSize, Size(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		imshow("KTL", image);

		// Exit the loop Esc key
		if (cv::waitKey(30) == 27) break;

		std::swap(points[1], points[0]);
		cv::swap(gray[1], gray[0]);
	}
}

// Function to compute the spatial gradients of an image
void KLTTracker::computeGradients(const Mat& image, Mat& Ix, Mat& Iy) {
	Sobel(image, Ix, CV_32F, 1, 0, 3);
	Sobel(image, Iy, CV_32F, 0, 1, 3);
	//CV_64F
}

// Function to calculate the optical flow using the Lucas-Kanade method
void KLTTracker::manualCalcOpticalFlowPyrLK(const Mat& prevImage, const Mat& nextImage,
	const vector<Point2f>& prevPoints, vector<Point2f>& nextPoints,
	vector<uchar>& status, vector<float>& error,
	Size winSize, int maxLevel, TermCriteria criteria, int flags,
	double minEigThreshold)
{
	nextPoints.resize(prevPoints.size());
	status.resize(prevPoints.size());
	error.resize(prevPoints.size());

	Mat Ix, Iy;
	computeGradients(prevImage, Ix, Iy);

	for (size_t i = 0; i < prevPoints.size(); i++) {
		Point2f pt = prevPoints[i];

		// Compute the window for each point
		Point2f window_tl(pt.x - winSize.width / 2, pt.y - winSize.height / 2);
		Point2f window_br(pt.x + winSize.width / 2, pt.y + winSize.height / 2);

		// Extract the region of interest for the window
		Mat IxROI = Ix(Rect(window_tl, window_br));
		Mat IyROI = Iy(Rect(window_tl, window_br));

		// Compute spatial gradient at the point
		float Ix_val = IxROI.at<float>(winSize.height / 2, winSize.width / 2);
		float Iy_val = IyROI.at<float>(winSize.height / 2, winSize.width / 2);

		// Compute temporal gradient at the point
		float It_val = static_cast<float>(nextImage.at<uchar>(pt) - prevImage.at<uchar>(pt));

		// Solve the system of linear equations for optical flow
		Mat A = (Mat_<float>(2, 2) << Ix_val, Iy_val, Iy_val, -Ix_val);
		Mat B = (Mat_<float>(2, 1) << -It_val, 0);

		Mat flow;
		solve(A, B, flow, DECOMP_SVD);

		// Update the next point
		nextPoints[i] = pt + Point2f(flow.at<float>(0, 0), flow.at<float>(1, 0));
		status[i] = 1;
		error[i] = norm(nextPoints[i] - pt);
	}

    //Ptr<cv::SparsePyrLKOpticalFlow> optflow = cv::SparsePyrLKOpticalFlow::create(winSize, maxLevel, criteria, flags, minEigThreshold);
	//optflow->calc(prevImage, nextImage, prevPoints, nextPoints, status, error);
}