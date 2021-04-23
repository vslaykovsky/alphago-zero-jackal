#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"
#include "tictactoe/tictactoe_train.h"
#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "tictactoe_train [--config json] [--config_file json_file]" << endl;
        exit(-1);
    }
    unordered_map<string, float> config;
    if (!strcmp(argv[1], "--config")) {
        config = load_config_from_string(argv[2]);
    } else if (!strcmp(argv[1], "--config_file")) {
        config = load_config_from_file(argv[2]);
    }
    std::string dir = "tmp/jackal";

    Jackal jackal(7, 7, 2);
    auto dims = jackal.get_state().sizes();
    JackalModel model(dims);
    JackalModel baseline_model(dims);
    int step;

    auto model_path = dir + "/model.bin";
    if (experimental::filesystem::exists(model_path)) {
        torch::load(model, model_path);
    }

    std::string ds_file;
    for(auto& p: filesystem::directory_iterator(dir)) {
        string s = p.path();
        if (s.rfind("selfplay") >= 0 && s > ds_file) {
            ds_file = s;
        }
    }
    SelfPlayDataset ds;
    ds.load(ds_file);
    Trainer<Jackal, JackalModel> trainer(config, torch::kCUDA);
    auto loss = trainer.train(model, ds, baseline_model, nullptr, step);
    torch::save(model, model_path);
}