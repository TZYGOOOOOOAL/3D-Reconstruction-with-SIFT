#include "utils.h"



// �ļ�����
bool is_exist(string path)
{
	return !bool(_access(path.c_str(), 0));
}

// ����1��Ŀ¼
bool make_dir(string dir_path)
{
	if (is_exist(dir_path) || is_file(dir_path))
		return false;
	_mkdir(dir_path.c_str());
	return true;
}

// ·��ƴ��
// https://blog.csdn.net/jiratao/article/details/9764679?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
string path_join(string path1, string path2)
{
	char path_buffer[_MAX_PATH];
	_makepath(path_buffer, NULL, path1.c_str(), path2.c_str(), NULL);
	return string(path_buffer);
}


static string my_split_path(const string &path, string mode)
{
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char _fname[_MAX_FNAME];
	char _ext[_MAX_EXT];

	_splitpath(path.c_str(), drive, dir, _fname, _ext);

	if (mode == "fname")
		return string(_fname);
	if (mode == "ext")
		return string(_ext);
	if (mode == "drive")
		return string(drive);
	if (mode == "dir")
		return string(drive) + dir;
	if (mode == "basename")
		return string(_fname) + _ext;
	return path;
}

// ��׺
string get_ext(string path){
	return my_split_path(path, "ext");
}

// �ļ���
string get_filename(string path){
	return my_split_path(path, "fname");
}

// basename
string get_basename(string path){
	return my_split_path(path, "basename");
}

// dir name
string get_dirname(string path){
	return my_split_path(path, "dir");
}

// ���ļ�
bool is_file(string path)
{
	return is_exist(path) && !get_ext(path).empty();
}

// ��Ŀ¼
bool is_dir(string path)
{
	return is_exist(path) && get_ext(path).empty();
}

// �����ļ�
vector<string> get_all_files(string path, vector<string> formats)
{
	vector<string> file_paths = get_child_files(path, formats);
	vector<string> dir_paths = get_child_dirs(path);

	// �ݹ����������Ŀ¼
	vector<string> temp_paths;
	for (int i = 0; i < dir_paths.size(); i++)
	{
		temp_paths = get_all_files(dir_paths[i], formats);
		file_paths.insert(file_paths.end(), temp_paths.begin(), temp_paths.end());
	}

	return file_paths;
}


// Ŀ¼���������ļ�
vector<string> get_child_files(string path, vector<string> formats)
{
	if (formats.empty())
		formats.emplace_back("");

	//�ļ����    
	intptr_t hFile = 0;
	vector<string> file_paths;

	//�ļ���Ϣ    
	struct _finddata_t fileinfo;
	string p;

	// ����
	for (int i = 0; i < formats.size(); i++)
	{
		string format = formats[i];
		if ((hFile = _findfirst(p.assign(path).append(string("/*") + format).c_str(), &fileinfo)) != -1)
		{
			do
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					file_paths.push_back(p.assign(path).append("/").append(fileinfo.name));

			} while (_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}
	}
	
	return file_paths;
}


vector<string> get_child_dirs(string path)
{
	//�ļ����    
	intptr_t  hFile = 0;
	vector<string> dir_paths;

	//�ļ���Ϣ    
	struct _finddata_t fileinfo;
	string p;

	// �����ļ���Ŀ¼
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			// ��Ŀ¼
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					dir_paths.emplace_back(p.assign(path).append("/").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}

	return dir_paths;
}

// ���������ֵ
void save_mat_data(Mat &m, string save_path)
{
	FileStorage fs(save_path, FileStorage::WRITE);
	fs << "data" << m;
}

Mat load_mat_data(string load_path)
{
	FileStorage fs(load_path, FileStorage::READ);
	Mat m;
	fs["data"] >> m;
	return m;
}

// ��ò��ظ�����·��
string get_no_repeat_save_path(string save_path)
{
	if (is_file(save_path))
		cout << "WARNING : save_path is Exist !!!" << endl;
	else
		return save_path;

	string dir = get_dirname(save_path);
	string ext = get_ext(save_path);
	string filename = get_filename(save_path);
	
	while (is_file(save_path))
	{
		save_path = path_join(dir, filename + "_" + to_string(rand()) + ext);
	}

	return save_path;
}

void get_left_right_imgs(Mat &img_left, Mat &img_right, const string& img_dir)
{
	vector<string> img_paths = get_child_files(img_dir);
	assert(img_paths.size() == 2);
	
	string first_img_filename = get_filename(img_paths[0]);
	char first_idx_char = first_img_filename[first_img_filename.size() - 1];

	if (first_idx_char == '2')
	{
		img_left = imread(img_paths[0], cv::IMREAD_UNCHANGED);
		img_right = imread(img_paths[1], cv::IMREAD_UNCHANGED);
	}
	else
	{
		img_left = imread(img_paths[1], cv::IMREAD_UNCHANGED);
		img_right = imread(img_paths[0], cv::IMREAD_UNCHANGED);
	}
	return;
}

// �õ���ֵ���������ݼ���ֵ��Ҫ���Թ̶������õ���ֵ
double get_gt_disparity_ratio(const string& path)
{
	string filename = get_filename(path);
	int str_len = filename.size();
	if (filename[str_len - 2] == '_' && filename[str_len - 1] == '2')
		return 4.0;
	return 8.0;
}

Timer::Timer()
{
	start_time = double(clock());
}

double Timer::get_run_time(string desc, bool reset, bool show)
{
	double run_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
	if (show)
		cout << desc + " run time = " << run_time << "(s)" << endl;
	if (reset)
		start_time = double(clock());
	return run_time;
}

void Timer::reset()
{
	start_time = double(clock());
	return;
}
