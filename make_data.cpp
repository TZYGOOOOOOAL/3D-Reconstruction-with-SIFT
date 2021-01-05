#include "make_data.h"
#include "utils.h"

// 处理原始pgm，ppm文件 -> bmp 文件
void make_data(const Config& config)
{
	vector<string> raw_dir_paths = get_child_dirs(config.RAW_DATA_DIR);

	for (int i = 0; i < raw_dir_paths.size(); i++)
	{
		// 路径配置
		string raw_dir_path = raw_dir_paths[i];
		string save_dir = path_join(config.DATA_DIR, get_filename(raw_dir_path));
		make_dir(save_dir);

		string img_save_dir = path_join(save_dir, config.IMG_REL_DIR);
		make_dir(img_save_dir);
		string gt_save_dir = path_join(save_dir, config.GT_REL_DIR);
		make_dir(gt_save_dir);

		// 读写图像文件
		vector<string> raw_img_paths = get_child_files(raw_dir_path, vector<string>({config.RAW_IMG_FMT}));
		for (int img_idx = 0; img_idx < raw_img_paths.size(); img_idx++)
		{
			string raw_img_filename = get_filename(raw_img_paths[img_idx]);
			char idx_char = raw_img_filename[raw_img_filename.size() - 1];
			if (idx_char != '2' && idx_char != '6')
				continue;
			Mat raw_img = cv::imread(raw_img_paths[img_idx], cv::IMREAD_UNCHANGED);
			string img_save_path = path_join(img_save_dir, raw_img_filename + config.IMG_FMT);
			imwrite(img_save_path, raw_img);
			cout << "save img " << img_save_path << endl;
			assert(is_file(img_save_path));
		}

		// 读写真值文件
		vector<string> raw_gt_paths = get_child_files(raw_dir_path, vector<string>({ config.RAW_GT_FMT }));
		for (int gt_idx = 0; gt_idx < raw_gt_paths.size(); gt_idx++)
		{
			string raw_gt_filename = get_filename(raw_gt_paths[gt_idx]);
			Mat raw_img = cv::imread(raw_gt_paths[gt_idx], cv::IMREAD_UNCHANGED);
			string gt_save_path = path_join(gt_save_dir, raw_gt_filename + config.IMG_FMT);
			imwrite(gt_save_path, raw_img);
			cout << "save gt " << gt_save_path << endl;
			assert(is_file(gt_save_path));
		}
	}
	
	return;
}