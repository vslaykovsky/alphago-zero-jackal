#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"

#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>

using namespace std;

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "tictactoe_train folder [--config json] [--config_file json_file]" << endl;
        exit(-1);
    }
    unordered_map<string, float> config;
    int argi = 1;
    string dir = argv[argi++];
    if (!strcmp(argv[argi], "--config")) {
        config = load_config_from_string(argv[argi + 1]);
    } else if (!strcmp(argv[argi], "--config_file")) {
        config = load_config_from_file(argv[argi + 1]);
    }

    int step = 0;


    Trainer<Jackal, JackalModel> trainer(config, torch::kCUDA);

    auto loss = trainer.train(dir, nullptr, step,
                              int(config["jackal_channels"]),
                              int(config["jackal_blocks"]),
                              int(config["jackal_players"]));

}