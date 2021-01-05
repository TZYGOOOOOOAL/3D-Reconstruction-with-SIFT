# ifndef __UTILS_H__
# define __UTILS_H__

#include <opencv2/opencv.hpp>
#include <direct.h>
#include <io.h>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

using namespace std;
using namespace cv;

class Timer{
private:
	double start_time;
public:
	Timer();
	double get_run_time(string desc="", bool reset=false, bool show=true);
	void reset();
};


bool is_exist(string path);
bool make_dir(string dir_path);
string path_join(string path1, string path2);
static string my_split_path(const string &path, string mode);
string get_ext(string path);
string get_filename(string path);
string get_basename(string path);
string get_dirname(string path);
bool is_file(string path);
bool is_dir(string path);

vector<string> get_all_files(string path, vector<string> formats = vector<string>());
vector<string> get_child_files(string path, vector<string> formats = vector<string>());
vector<string> get_child_dirs(string path);
void save_mat_data(Mat &m, string save_path);
Mat load_mat_data(string load_path);

string get_no_repeat_save_path(string save_path);

void get_left_right_imgs(Mat &img_left, Mat &img_right, const string& img_dir);
double get_gt_disparity_ratio(const string& path);
# endif