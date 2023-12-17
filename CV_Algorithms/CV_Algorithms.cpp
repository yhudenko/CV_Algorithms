#include <opencv2/opencv.hpp>
#include <iostream>
#include "CornerDetectorFAST.h"
#include "KLTTracker.h"

using namespace cv;

int main()
{
    const std::string sourcePath = "Resources/Input";
    const std::string targetPath = "Resources/Output";
    const std::string fileName = "video.mp4";

    /*cv::Mat inputImage = cv::imread(sourcePath + "/" + fileName, IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Create a FAST detector object
    Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create(40, true, cv::FastFeatureDetector::TYPE_9_16);

    // Detect FAST corners
    std::vector<KeyPoint> keypoints;
    fastDetector->detect(inputImage, keypoints);

    // Draw keypoints on the image
    Mat outputImage;
    cv::drawKeypoints(inputImage, keypoints, outputImage, Scalar(0, 255, 0));

    // Display the input and output images
    imshow("Input Image", inputImage);
    imshow("FAST Corners", outputImage);

    CornerDetectorFAST detector = CornerDetectorFAST(sourcePath + "/" + fileName);
    const cv::Mat manualOutputImage = detector.detectFAST(70);
    imshow("Manual FAST Output", manualOutputImage);

    //Save images
    imwrite(targetPath + "/" + fileName, outputImage);
    imwrite(targetPath + "/manual" + fileName, manualOutputImage);*/

    KLTTracker tracker = KLTTracker(sourcePath + "/" + fileName);
    tracker.showTrack();

    waitKey(0);

    cv::destroyAllWindows();
    return 0;
}