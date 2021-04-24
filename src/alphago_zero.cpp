#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"
#include "tictactoe/tictactoe_train.h"
#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>
#include "util/utils.h"

using namespace std;

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "alphago_zero game dir [--config json] [--config_file json_file]" << endl;
        exit(-1);
    }
    int argi = 1;
    std::string game = argv[argi++];
    std::string dir = argv[argi++];
    unordered_map<string, float> config_map;
    if (!strcmp(argv[argi], "--config")) {
        config_map = load_config_from_string(argv[argi+1]);
    } else if (!strcmp(argv[argi], "--config_file")) {
        config_map = load_config_from_file(argv[argi+1]);
    }
    if (game == "tictactoe") {
        cout << tictactoe_train(dir, config_map) << endl;
    } else {
        cout << jackal_train(dir, config_map) << endl;
    }
}