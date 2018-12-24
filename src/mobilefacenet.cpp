#include "mobilefacenet.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace cv;
using namespace std;

Recognize::Recognize(const std::string &model_path) {
	std::string param_files = model_path + "/mobilefacenet.param";
	std::string bin_files = model_path + "/mobilefacenet.bin";
	Recognet.load_param(param_files.c_str());
	Recognet.load_model(bin_files.c_str());
}

Recognize::~Recognize() {
	Recognet.clear();
}

void Recognize::RecogNet(ncnn::Mat& img_, std::vector<float> &feature) {
	ncnn::Extractor ex = Recognet.create_extractor();
	ex.set_num_threads(2);
	ex.set_light_mode(true);
	ex.input("data", img_);
	ncnn::Mat out;
	ex.extract("fc1", out);
    feature.resize(128);
    for (int j = 0; j < 128; j++)
		feature[j] = out[j];

    normalize(feature);
}

void Recognize::normalize(std::vector<float> &feature)
{
    float sum = 0.f;
    for(auto it = feature.begin(); it != feature.end(); ++it)
        sum += (*it) * (*it);

    sum = sqrt(sum);
    for(auto it = feature.begin(); it != feature.end(); ++it)
        *it /= sum;
}

void Recognize::start(const cv::Mat& img, std::vector<float>&feature) {
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
	RecogNet(ncnn_img, feature);
}

float calculSimilar(std::vector<float>& v1, std::vector<float>& v2)
{
	float sum = 0.f;
	for(int i = 0; i < v1.size(); ++i)
		sum += v1[i] * v2[i];

	return sum;
}
