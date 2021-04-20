#pragma once

#include <torch/torch.h>

#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <random>
#include <boost/functional/hash.hpp>


typedef cv::Point Coords;

namespace std {
    template<>
    struct hash<Coords> {
        std::size_t operator()(const Coords &action) const {
            using boost::hash_value;
            using boost::hash_combine;
            std::size_t seed = 0;
            hash_combine(seed, hash_value(action.x));
            hash_combine(seed, hash_value(action.y));
            return seed;
        }
    };
}


const int TILE_SIZE = 128;

float rand01();
std::string to_string(const torch::Tensor &t);


void copy_with_alpha(cv::Mat& to, cv::Mat& from, int xPos, int yPos);

inline cv::Point tile_center(const Coords& p) {
    return {int((p.x + 0.5) * TILE_SIZE), int((p.y + 0.5) * TILE_SIZE)};
};

template<class T1, class T2>
inline std::ostream& operator<<(std::ostream&os, const std::unordered_map<T1, T2>& v) {
    for (auto& kv: v) {
        os << kv.first << ": " << kv.second << std::endl;
    }
    return os;
}




std::default_random_engine& get_generator();
