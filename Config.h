# ifndef __CONFIG_H__
# define __CONFIG_H__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct Config
{
	/****** ���� ******/
	string DATA_DIR = "data/processed_data";
	string IMG_REL_DIR = "img";
	string GT_REL_DIR = "ground_truth";
	string IMG_FMT = ".bmp";

	string RAW_DATA_DIR = "data/raw_data";
	string RAW_GT_FMT = ".pgm";
	string RAW_IMG_FMT = ".ppm";

	/****** SIFT���� ******/
	int SIFT_KEYPOINTS_KEEP_NUM = 0;	// ����ȫ��������
	int SIFT_LAYERS_NUM = 3;			// ��������ÿ�������default=3��
	double SIFT_CONTRAST_TH = 0.02;		// ���˵��ϲ������㣨0.04��
	double SIFT_EDGE_TH = 15.0;			// ���˵���ԵЧӦ����ֵ��Խ�󱻹��˵�Խ�٣�10.0��
	double SIFT_SIGMA = 1.6;			// ��0��sigma��1.6��

	/****** ƥ����� ******/
	float MATCH_DIST_RATIO = 0.75f;			// ԽС�õ�ƥ��Խ�٣�Խ����ƥ���Խ�ࣨ0.6��
	float MATCH_MIN_DIST_TIMES_TH = 5.0f;	// ƥ��������ֵ �� ��С����Ķ��ٱ���5.0��
	
	/****** ���ӻ� ******/
	bool VISUAL_TEMP_RESULT = true;
	Scalar VISUAL_PREDICT_BBOX_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_PREDICT_MASK_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_TARGET_BBOX_COLOR = Scalar(255, 0, 0);

	/****** ���� ******/
	double EVAL_PIXEL_DIFF_TH = 20;
	double EVAL_IOU_TH = 0.5;

	/****** Ԥ�� ******/
	bool TEST_WITH_LABEL = true;
	bool SAVE_RESULT = false;
	string SAVE_DIR_PATH = "result";
};

# endif