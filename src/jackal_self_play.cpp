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
    unordered_map<string, float> config_map;
    if (!strcmp(argv[2], "--config")) {
        config_map = load_config_from_string(argv[3]);
    } else if (!strcmp(argv[2], "--config_file")) {
        config_map = load_config_from_file(argv[3]);
    }
    multithreaded_self_plays(7, 7, 2, model, config);
}