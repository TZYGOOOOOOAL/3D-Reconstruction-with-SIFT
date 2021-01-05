#include "disparity.h"

// https://blog.csdn.net/BrookIcv/article/details/79069017
void compute_disparity_d2c(Mat& img_left, Mat& img_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right,
	Mat& disparity, const Config& config)
{
	// KeyPoint -> pts
	vector<cv::Point2f> pts_left, pts_right;
	cv::KeyPoint::convert(key_points_left, pts_left);
	cv::KeyPoint::convert(key_points_right, pts_right);

	// 计算单映射矩阵，由于没有相机内参只能近似计算视差
	// 在平移相对景深较小的情况下可用单映射矩阵近似变换矩阵
	Mat H1, H2;
	H1 = cv::findHomography(pts_left, pts_right);
	H2 = cv::findHomography(pts_right, pts_left);
	
	// 计算视差
	double std_one;
	disparity = Mat(img_left.size(), CV_64FC1);

	double H_data[9];
	memcpy(H_data, (double*)(H1.data), 9 * sizeof(double));

	for (int y1 = 0; y1 < img_left.rows; y1++)
	{
		for (int x1 = 0; x1 < img_left.cols; x1++)
		{
			// 只用计算x
			double x2 = H_data[0] * x1 + H_data[1] * y1 + H_data[2];
			double y2 = H_data[3] * x1 + H_data[4] * y1 + H_data[5];
			std_one = H_data[6] * x1 + H_data[7] * y1 + H_data[8];
			
			x2 /= std_one;
			y2 /= std_one;

			disparity.at<double>(y1, x1) = x1 - x2;
		}
	}

	// 可视化
	if (config.VISUAL_TEMP_RESULT)
	{
		Mat visual_disparity;
		cv::normalize(disparity, visual_disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		imshow("disparity_d2c", visual_disparity);
		waitKey();
		cv::destroyWindow("disparity_d2c");
	}
}


// 离散值
void compute_disparity_d(Mat& img_left, Mat& img_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right,
	vector<double>& disparity_vec, const Config& config)
{
	disparity_vec.clear();
	for (int i = 0; i < key_points_left.size(); i++)
	{
		KeyPoint kp_left = key_points_left[i];
		KeyPoint kp_right = key_points_right[i];
		disparity_vec.push_back(kp_left.pt.x - kp_right.pt.x);
	}
}


// 连续值
void compute_disparity_c(Mat& img_left, Mat& img_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right,
	Mat& disparity, Mat& match_mask, const Config& config)//
{
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 11);
	sgbm->compute(img_left, img_right, disparity);
	match_mask = disparity >= 0;
	disparity.convertTo(disparity, CV_64FC1, 1/16.0);		// /16得到真实视差

	// 可视化
	if (config.VISUAL_TEMP_RESULT)
	{
		Mat visual_disparity_left;
		Mat visual_disparity_right;
		cv::normalize(disparity, visual_disparity_left, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//disp.convertTo(visual_disparity_left, CV_8U, 255 / (16 *16.));
		imshow("disparity_c", visual_disparity_left);
		waitKey();
		cv::destroyWindow("disparity_c");
	}
}