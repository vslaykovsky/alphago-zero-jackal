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
    JackalModel model(dims);
    auto model_path = dir + "/model.bin";
    if (experimental::filesystem::exists(model_path)) {
        torch::load(model, model_path);
        cout << "Loaded model from " << model_path << endl;
    }
    auto selfplay_files = get_selfplay_files(dir);
    const string &selfplay_file = dir + "/selfplay_" + std::to_string(selfplay_files.size()) + ".bin";
    cout << "Running selfplays and saving to " << selfplay_file << endl;
    srand(123);
    std::vector<SelfPlayResult> self_plays = multithreaded_self_plays((int) config["jackal_height"],
                                                                      (int) config["jackal_width"],
                                                                      (int) config["jackal_players"],
                                                                      model,
                                                                      config);
    SelfPlayDataset ds(self_plays, (int) config["train_batch_size"]);
    ds.save(selfplay_file);
}