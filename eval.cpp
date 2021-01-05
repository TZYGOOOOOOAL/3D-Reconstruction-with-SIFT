#include "eval.h"

// https://blog.csdn.net/danmeng8068/article/details/103045501/?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-2&spm=1001.2101.3001.4242
void eval(Mat& disparity, Mat& gt, Mat& mask, const double& ratio)
{
	Mat gt_d;
	gt.convertTo(gt_d, CV_64FC1, 1. / ratio);

	// 误差方均根
	double root_mean_squared;
	Mat diff_mat;
	cv::subtract(disparity, gt_d, diff_mat, mask);
	cv::pow(diff_mat, 2, diff_mat);
	root_mean_squared = cv::sum(diff_mat)[0];
	root_mean_squared /= disparity.rows* disparity.cols;
	root_mean_squared = std::sqrt(root_mean_squared);

	cout << "*** EVAL Result ***" << endl;
	cout << "RMS = " << root_mean_squared << endl;
	return;
}

// 重载SIFT离散点评价
void eval(vector<double>& disparity_vec, vector<KeyPoint> &key_pts, Mat& gt, const double& ratio)
{
	Mat gt_d;
	gt.convertTo(gt_d, CV_64FC1, 1. / ratio);

	// 误差方均根
	double root_mean_squared;
	double error_sum = 0.0;
	double err;
	double gt_val, predict_val;
	int x, y;
	int num = disparity_vec.size();

	// 计算每一点误差
	for (int i = 0; i < num; i++)
	{
		predict_val = disparity_vec[i];
		x = cvRound(key_pts[i].pt.x);
		y = cvRound(key_pts[i].pt.y);
		gt_val = gt_d.at<double>(y, x);

		err = gt_val - predict_val;
		err *= err;

		error_sum += err;
	}
	
	error_sum /= num;
	root_mean_squared = std::sqrt(error_sum);

	cout << "*** EVAL Result ***" << endl;
	cout << "RMS = " << root_mean_squared << endl;
	return;
}