#include <opencv2/opencv.hpp>  //ͷ�ļ�
#include <opencv2/xfeatures2d.hpp>
#include "Config.h"
#include "make_data.h"
#include "utils.h"
#include "feature_match.h"
#include "disparity.h"
#include "eval.h"

using namespace cv;  //����cv�����ռ�
using namespace std;

int main()
{
	Config config;
	
	Mat img_left, img_right;
	Mat gt_left, gt_right;
	Mat descriptors_left, descriptors_right;
	vector<cv::KeyPoint> key_points_left, key_points_right;
	vector<cv::DMatch> matches;

	Mat disparity;
	vector<double> disparity_vec;
	Mat disparity_mask;
	double disparity_ratio;

	string dir_path, img_dir, gt_dir;

	// ��������
	//make_data(config);
	vector<string> dir_paths = get_child_dirs(config.DATA_DIR);

	for (int i = 0; i < dir_paths.size(); i++)
	{
		dir_path = dir_paths[i];
		img_dir = path_join(dir_path, config.IMG_REL_DIR);
		gt_dir = path_join(dir_path, config.GT_REL_DIR);

		cout << "\nProcess in " << get_filename(dir_path) << endl;

		// ԭͼ��
		get_left_right_imgs(img_left, img_right, img_dir);

		// ��ֵͼ��
		get_left_right_imgs(gt_left, gt_right, gt_dir);

		// ��ȡ������ƥ��
		extract_and_match_features(img_left, img_right, descriptors_left, descriptors_right,
			key_points_left, key_points_right, matches, config);

		// �����Ӳ�
		//compute_disparity_d2c(img_left, img_right, key_points_left, key_points_right, disparity, config);
		compute_disparity_c(img_left, img_right, key_points_left, key_points_right, disparity, disparity_mask, config);
		compute_disparity_d(img_left, img_right, key_points_left, key_points_right, disparity_vec, config);

		// ��֤
		disparity_ratio = get_gt_disparity_ratio(dir_path);
		eval(disparity, gt_left, disparity_mask, disparity_ratio);
		eval(disparity_vec, key_points_left, gt_left, disparity_ratio);

	}

	system("pause");
}