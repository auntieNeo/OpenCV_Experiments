#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(argv[1]);

  std::cout << "Reading the images...";
  std::cout.flush();
  cv::Mat img1 = cv::imread( argv[2] );
//  cv::Mat img2 = cv::imread( argv[3] );

  std::cout << " Done." << std::endl;


  cv::vector<cv::KeyPoint> keypoints1;
  detector->detect(img1, keypoints1);
  std::cout << "Found " << keypoints1.size() << " keypoints." << std::endl;
  cv::Mat outImage;
  cv::drawKeypoints(img1, keypoints1, outImage);
  std::cout << "Writing to file: " << argv[3] << std::endl;
  cv::imwrite(argv[3], outImage);
  return 0;
}
