#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"
#include "tictactoe/tictactoe_train.h"
#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "tictactoe_train game [--config json] [--config_file json_file]" << endl;
        exit(-1);
    }
    string game = argv[1];
    string config;
    if (!strcmp(argv[2], "--config")) {
        config = argv[3];
    } else if (!strcmp(argv[2], "--config_file")) {
        string json_path = argv[3];
        std::ifstream t(json_path);
        std::stringstream buffer;
        buffer << t.rdbuf();
        config = (buffer.str());
    }
    auto json_config = json::parse(config);

    unordered_map<string, float> config_map;
    for (json::iterator it = json_config.begin(); it != json_config.end(); ++it) {
        config_map[it.key()] = it.value();
    }
    if (game == "tictactoe") {
        cout << tictactoe_train(config_map) << endl;
    } else {
        cout << jackal_train(config_map) << endl;
    }
}