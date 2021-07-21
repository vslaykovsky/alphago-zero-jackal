#include "tictactoe/tictactoe.h"
#include "tictactoe/tictactoe_model.h"
#include "rl/self_play.h"
#include "rl/train.h"

#include "jackal/jackal_train.h"
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;


Jackal gen_state(
        unordered_map<string, float> config,
        int col) {
    int height = (int) config["jackal_height"];
    int width = (int) config["jackal_width"];
    int players = (int) config["jackal_players"];
    Jackal jackal(height, width, players, true, true);
    auto dims = jackal.get_state().sizes();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (jackal.ground.is_ground(x, y)) {
                jackal.ground.set_ground(x, y);
                jackal.ground.set_gold(x, y, x == col && y == 3? 5 : 0);
            }
        }
    }
    return jackal;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "tictactoe_train model [--config json] [--config_file json_file]" << endl;
        exit(-1);
    }
    unordered_map<string, float> config;
    int argi = 1;
    string model_path = argv[argi++];
    if (!strcmp(argv[argi], "--config")) {
        config = load_config_from_string(argv[argi + 1]);
    } else if (!strcmp(argv[argi], "--config_file")) {
        config = load_config_from_file(argv[argi + 1]);
    }

    for (int col = 1; col < 7 - 1; col++) {
        auto jackal = gen_state(config, col);
        JackalModel model(jackal.get_state().sizes(),
                          int(config["jackal_channels"]),
                          int(config["jackal_blocks"]),
                          int(config["jackal_players"]),
                          config["enable_action_value"] > 0);
        torch::load(model, model_path);
        GameModelOutput out = model(jackal.get_state().to(torch::kCUDA));
        auto sav = to_state_action_value(out, jackal);
        cv::imwrite("tmp/jackal_model_test/col" + to_string(col) + ".png",
                    jackal.get_image(&sav));
        cout << "col " << col << " state value " << out.value << endl;
        cout << jackal.ground.state[PLANE_GOLD] << endl;
    }
}