#include "utils.h"

using namespace cv;
using namespace std;

float rand01() {
    return rand() / (float) RAND_MAX;
}

std::string to_string(const torch::Tensor &t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

void copy_with_alpha(cv::Mat &to, cv::Mat &from, int xPos, int yPos) {
    Mat mask;
    vector<Mat> layers;

    split(from, layers); // seperate channels
    Mat rgb[3] = {layers[0], layers[1], layers[2]};
    mask = layers[3]; // png's alpha channel used as mask
    Mat rgb_mat;
    merge(rgb, 3, rgb_mat);  // put together the RGB channels, now from insn't transparent
    rgb_mat.copyTo(to.rowRange(yPos, yPos + rgb_mat.rows).colRange(xPos, xPos + rgb_mat.cols), mask);
}

std::default_random_engine dre;

std::default_random_engine& get_generator() {
    return dre;
}
