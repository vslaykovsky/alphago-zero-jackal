#include <gtest/gtest.h>
#include "train.h"
#include "play.h"

TensorBoardLogger gen_logger() {
    std::time_t t = std::time(nullptr);
    char mbstr[100];
    std::strftime(mbstr, sizeof(mbstr), "%Y-%m-%d_%H_%M_%S", std::localtime(&t));
    auto dir = "runs/run-" + std::string(mbstr) + "__" + std::to_string(rand() % 1000);
    std::experimental::filesystem::create_directories(dir);
    return TensorBoardLogger((dir + "/tfevents.pb").c_str());
}
