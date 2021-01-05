#ifndef __EVAL_H__
#define __EVAL_H__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Config.h"

using namespace cv;
using namespace std;

void eval(Mat& disparity, Mat& gt, Mat& mask, const double& ratio);
void eval(vector<double>& disparity_vec, vector<KeyPoint> &key_pts, Mat& gt, const double& ratio);
#endif
