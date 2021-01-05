# ifndef __CONFIG_H__
# define __CONFIG_H__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct Config
{
	/****** 数据 ******/
	string DATA_DIR = "data/processed_data";
	string IMG_REL_DIR = "img";
	string GT_REL_DIR = "ground_truth";
	string IMG_FMT = ".bmp";

	string RAW_DATA_DIR = "data/raw_data";
	string RAW_GT_FMT = ".pgm";
	string RAW_IMG_FMT = ".ppm";

	/****** SIFT参数 ******/
	int SIFT_KEYPOINTS_KEEP_NUM = 0;	// 保留全部特征点
	int SIFT_LAYERS_NUM = 3;			// 金字塔中每组层数（default=3）
	double SIFT_CONTRAST_TH = 0.02;		// 过滤掉较差特征点（0.04）
	double SIFT_EDGE_TH = 15.0;			// 过滤掉边缘效应的阈值，越大被过滤掉越少（10.0）
	double SIFT_SIGMA = 1.6;			// 第0层sigma（1.6）

	/****** 匹配参数 ******/
	float MATCH_DIST_RATIO = 0.75f;			// 越小得到匹配越少，越大误匹配对越多（0.6）
	float MATCH_MIN_DIST_TIMES_TH = 5.0f;	// 匹配距离最大值 是 最小距离的多少倍（5.0）
	
	/****** 可视化 ******/
	bool VISUAL_TEMP_RESULT = true;
	Scalar VISUAL_PREDICT_BBOX_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_PREDICT_MASK_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_TARGET_BBOX_COLOR = Scalar(255, 0, 0);

	/****** 评价 ******/
	double EVAL_PIXEL_DIFF_TH = 20;
	double EVAL_IOU_TH = 0.5;

	/****** 预测 ******/
	bool TEST_WITH_LABEL = true;
	bool SAVE_RESULT = false;
	string SAVE_DIR_PATH = "result";
};

# endif