#ifndef __DISPARITY_H__
#define __DISPARITY_H__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Config.h"

using namespace cv;
using namespace std;

void compute_disparity_d2c(Mat& img_left, Mat& img_right, vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, Mat& disparity, const Config& config);
void compute_disparity_d(Mat& img_left, Mat& img_right, vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, vector<double>& disparity_vec, const Config& config);
void compute_disparity_c(Mat& img_left, Mat& img_right, vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, Mat& disparity, Mat& match_mask, const Config& config);
#endif
