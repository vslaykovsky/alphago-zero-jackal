#include "utils.h"
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;


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


// root only
void set_high_thread_priority() {
    int priority_max = sched_get_priority_max(SCHED_FIFO);
    int priority_min = sched_get_priority_min(SCHED_FIFO);
    pthread_t main_id = pthread_self();
    struct sched_param param{};
    param.sched_priority=priority_max;
    int status = pthread_setschedparam(main_id, SCHED_FIFO, &param);
    if (status != 0)
        perror("pthread_setschedparam");
}



std::unordered_map<std::string, float> load_config_from_string(const std::string& config) {
    auto json_config = json::parse(config);
    unordered_map<string, float> config_map;
    for (json::iterator it = json_config.begin(); it != json_config.end(); ++it) {
        config_map[it.key()] = it.value();
    }
    return config_map;
}

std::unordered_map<std::string, float> load_config_from_file(const std::string& fname) {
    std::ifstream t(fname);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return load_config_from_string(buffer.str());
}
