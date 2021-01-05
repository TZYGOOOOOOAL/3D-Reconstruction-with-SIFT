#include "feature_match.h"

Mat extract_and_match_features(
	Mat& img_left, Mat& img_right,
	Mat& descriptors_left, Mat& descriptors_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right,
	vector<cv::DMatch>& matches,
	const Config& config
	)
{
	extract_features(img_left, img_right, descriptors_left, descriptors_right, key_points_left,
		key_points_right, config);
	Mat mat_Fundamental = match_features(descriptors_left, descriptors_right, matches, key_points_left, key_points_right, config);
	
	// ���ӻ�
	if (config.VISUAL_TEMP_RESULT)
	{
		Mat visual_left, visual_right;
		cv::drawKeypoints(img_left, key_points_left, visual_left);
		cv::drawKeypoints(img_right, key_points_right, visual_right);
		imshow("left key points", visual_left);
		imshow("right key points", visual_right);
		waitKey();
		cv::destroyWindow("left key points");
		cv::destroyWindow("right key points");

		Mat visual_match;
		cv::drawMatches(img_left, key_points_left, img_right, key_points_right, matches, visual_match);
		imshow("match result", visual_match);
		waitKey();
		cv::destroyWindow("match result");
	}

	// ���ػ�������
	return mat_Fundamental;
}


void extract_features(
	Mat& img_left, Mat& img_right,
	Mat& descriptors_left, Mat& descriptors_right,
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right,
	const Config& config
	)
{
	key_points_left.clear();
	key_points_right.clear();

	// SIFT 
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create( config.SIFT_KEYPOINTS_KEEP_NUM,
		config.SIFT_LAYERS_NUM, config.SIFT_CONTRAST_TH, config.SIFT_EDGE_TH, config.SIFT_SIGMA);

	// ����ͼsift������ȡ
	sift->detectAndCompute(img_left, cv::noArray(), key_points_left, descriptors_left);

	// ����ͼsift������ȡ
	sift->detectAndCompute(img_right, cv::noArray(), key_points_right, descriptors_right);

	return;
}


// *********** ƥ��������������ƥ�� ***********
cv::Mat match_features(Mat& descriptors_left, Mat& descriptors_right, vector<cv::DMatch>& matches, 
	vector<cv::KeyPoint>& key_points_left, vector<cv::KeyPoint>& key_points_right, const Config& config)
{
	matches.clear();

	// FLANN ƥ���㷨��KD��ʵ�֣� ԭ�� http://www.whudj.cn/?p=920   �����滻Ϊcv::BFMatcher���б���ƥ��
	cv::FlannBasedMatcher matcher;
	vector<vector<cv::DMatch>> knn_matches;

	// k-���ڣ�k=2��������k������ΪҪ�Ƚ�����ں͵�2���ڱ�ֵ����ɸѡ
	matcher.knnMatch(descriptors_left, descriptors_right, knn_matches, 2);

	// ɸѡ
	// Step 1��Ratio Test
	float min_dist = FLT_MAX;
	float best_dist;
	for (int i = 0; i < knn_matches.size(); i++)
	{
		// �����/����� ����
		best_dist = knn_matches[i][0].distance;
		if (best_dist > 0.6 * knn_matches[i][1].distance)
			continue;

		// ��¼��Сֵ
		min_dist = std::min(min_dist, best_dist);
	}

	for (int i = 0; i < knn_matches.size(); ++i)
	{
		//�ų�������Ratio Test�ĵ��ƥ��������ĵ�
		best_dist = knn_matches[i][0].distance;
		if ( best_dist > 0.6 * knn_matches[i][1].distance || best_dist > 3 * min_dist)
			continue;

		//����ƥ���
		matches.push_back(knn_matches[i][0]);
	}

	// Step 2: RANSAC     
	// https://blog.csdn.net/u011867581/article/details/24051885?utm_medium=distribute.pc_relevant_bbs_down.none-task-blog-baidujs-2.nonecase&depth_1-utm_source=distribute.pc_relevant_bbs_down.none-task-blog-baidujs-2.nonecase
	int pts_num = (int)matches.size();
	Mat p1(pts_num, 2, CV_32F);
	Mat p2(pts_num, 2, CV_32F);

	// ��Keypointת��ΪMat
	Point2f pt;
	for (int i = 0; i < pts_num; i++)
	{
		pt = key_points_left[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = key_points_right[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// ��RANSAC���������������F��ֻ�ܵõ� ���뼶�߹�ϵ
	Mat mat_Fundamental;
	vector<uchar> RANSACStatus;		// ���ڴ洢RANSAC��ÿ�����״̬

	mat_Fundamental = findFundamentalMat(p1, p2, RANSACStatus, cv::FM_RANSAC);

	// ����������
	int outliner_num = 0;
	for (int i = 0; i < pts_num; i++)
	{
		if (RANSACStatus[i] == 0) // ״̬Ϊ0��ʾ���
			outliner_num++;
	}

	// �����ڵ㣬���ڱ����ڵ��ƥ���ϵ
	int inliner_num = pts_num - outliner_num;
	vector<Point2f> inliner_pts_left(inliner_num);
	vector<Point2f> inliner_pts_right(inliner_num);
	vector<DMatch> inlier_matches(inliner_num);

	cout << "After RANSAC : inliner pts= " << inliner_num << " / outliner pts= " << outliner_num << endl;

	int inliner_idx = 0;
	for (int i = 0; i < pts_num; i++)
	{
		if (RANSACStatus[i] != 0)
		{
			inliner_pts_left[inliner_idx].x = p1.at<float>(i, 0);
			inliner_pts_left[inliner_idx].y = p1.at<float>(i, 1);
			inliner_pts_right[inliner_idx].x = p2.at<float>(i, 0);
			inliner_pts_right[inliner_idx].y = p2.at<float>(i, 1);
			inlier_matches[inliner_idx].queryIdx = inliner_idx;
			inlier_matches[inliner_idx].trainIdx = inliner_idx;
			inliner_idx++;
		}
	}

	// ���ڵ�ת�� 
	vector<KeyPoint> key1(inliner_num);
	vector<KeyPoint> key2(inliner_num);
	KeyPoint::convert(inliner_pts_left, key1);
	KeyPoint::convert(inliner_pts_right, key2);

	key_points_left = key1;
	key_points_right = key2;
	matches = inlier_matches;

	return mat_Fundamental;
}