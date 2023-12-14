#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main()
{
    const std::string sourcePath = "Resources/Input";
    const std::string targetPath = "Resources/Output";
    const std::string fileName = "room2.png";
    // Read the input image
    cv::Mat inputImage = cv::imread(sourcePath + "/" + fileName, IMREAD_GRAYSCALE);

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


    imwrite(targetPath + "/" + fileName, outputImage);
    // Display the input and output images
    //cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
    imshow("Input Image", inputImage);
    //cv::namedWindow("FAST Corners", cv::WINDOW_NORMAL);
    imshow("FAST Corners", outputImage);
    waitKey(0);

    return 0;
}