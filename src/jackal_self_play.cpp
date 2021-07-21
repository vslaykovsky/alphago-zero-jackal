#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"

#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "tictactoe_train game [--config json] [--config_file json_file]" << endl;
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
    Jackal jackal((int) config["jackal_height"], (int) config["jackal_width"], (int) config["jackal_players"]);
    auto dims = jackal.get_state().sizes();
    JackalModel model(dims,
                      int(config["jackal_channels"]),
                      int(config["jackal_blocks"]),
                      int(config["jackal_players"]),
                      config["enable_action_value"] > 0);
    auto model_path = dir + "/model.bin";
    if (experimental::filesystem::exists(model_path)) {
        cout << "Loading model from " << model_path << endl;
        torch::load(model, model_path);
    } else {
        throw std::runtime_error("no model.bin found");
    }
    cout << model << endl;
    auto selfplay_files = get_selfplay_files(dir);
    cout << "Running selfplays and saving to " << dir << endl;
    time_t tm;
    time(&tm);
    srand(tm);
    if (selfplay_files.empty()) {
        config["mcts_iterations"] = config["mcts_iterations_first_cycle"];
    }
    for (auto &kv: config) {
        std::cout << "  " << kv.first << ": \t" << kv.second << std::endl;
    }
    multithreaded_self_plays(
            dir,
            (int) config["jackal_width"], (int) config["jackal_height"],
            model,
            config,
            (int) config["jackal_players"]);
}