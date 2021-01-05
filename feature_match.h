#ifndef __FEATURE_MATCH_H__
#define __FEATURE_MATCH_H__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Config.h"

using namespace cv;
using namespace std;

Mat extract_and_match_features(Mat& img_left, Mat& img_right, Mat& descriptors_left, Mat& descriptors_right, vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, vector<cv::DMatch>& matches, const Config& config);
void extract_features(Mat& img_left, Mat& img_right, Mat& descriptors_left, Mat& descriptors_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, const Config& config);
Mat match_features(Mat& descriptors_left, Mat& descriptors_right, vector<cv::DMatch>& matches, vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, const Config& config);
#endif
