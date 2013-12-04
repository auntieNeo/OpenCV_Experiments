#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

cv::vector<cv::KeyPoint> computeKeypoints(cv::Mat image, cv::Ptr<cv::FeatureDetector> detector)
{
  cv::vector<cv::KeyPoint> keypoints;
  detector->detect(image, keypoints);
  std::cout << "Found " << keypoints.size() << " keypoints." << std::endl;
  return keypoints;
}

cv::Mat computeDescriptors(cv::Mat image, cv::vector<cv::KeyPoint> keypoints, cv::Ptr<cv::DescriptorExtractor> extractor)
{
  cv::Mat descriptors;
  extractor->compute(image, keypoints, descriptors);
  return descriptors;
}

void listMatches(cv::vector<cv::DMatch> matches)
{
  for(size_t i = 0; i < matches.size(); ++i)
  {
    std::cout << "Match distance: " << matches[i].distance << std::endl;
  }
}

int main(int argc, char **argv)
{
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(argv[1]);
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(argv[2]);
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(argv[3]);

  cv::Mat tmp_image;

  std::cout << "Reading the images...";
  std::cout.flush();
  cv::Mat image1 = cv::imread( argv[4] );
  cv::Mat image2 = cv::imread( argv[5] );

  std::cout << " Done." << std::endl;

  // compute the keypoints for each image
  cv::vector<cv::KeyPoint> keypoints1 = computeKeypoints(image1, detector);
  cv::vector<cv::KeyPoint> keypoints2 = computeKeypoints(image2, detector);

  // compute the descriptors for the keypoints
  cv::Mat descriptors1;
  extractor->compute(image1, keypoints1, descriptors1);
  cv::drawKeypoints(image1, keypoints1, tmp_image);
  cv::imwrite("/tmp/keypoints1.jpg", tmp_image);
  cv::Mat descriptors2;
  extractor->compute(image2, keypoints2, descriptors2);
  cv::drawKeypoints(image2, keypoints2, tmp_image);
  cv::imwrite("/tmp/keypoints2.jpg", tmp_image);
//  cv::Mat descriptors1 = computeDescriptors(image1, keypoints1, extractor);
//  cv::Mat descriptors2 = computeDescriptors(image2, keypoints2, extractor);

  // matching descriptors
  cv::vector<cv::DMatch> matches;
  matcher->match(descriptors1, descriptors2, matches);
  listMatches(matches);

  // drawing the results
  cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, tmp_image);
  cv::imwrite("/tmp/matches.jpg", tmp_image);

  return 0;
}
