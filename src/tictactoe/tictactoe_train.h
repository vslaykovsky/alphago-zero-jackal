#include <nlohmann/json.hpp>
#include "tictactoe.h"
#include "tictactoe_model.h"
#include "../rl/train.h"

float tictactoe_train(std::unordered_map<std::string, float> config_map) {
    TicTacToeModel model;
    TicTacToeModel baseline_model;

    std::unordered_map<std::string, float> default_config{
            {"train_learning_rate",     0.00012508178083968708},
            {"train_l2_regularization", 1.2820642389635053e-06},
            {"train_replay_buffer",     2048},
            {"train_epochs",            4},
            {"train_batch_size",        32},
            {"simulation_cycle_games",  64},
            {"simulation_cycles",       1000},
            {"simulation_temperature",  0.32326465799113663},
            {"mcts_iterations",         256},
            {"mcts_exploration",        2.0039111443673305},
            {"eval_size",               200},
            {"eval_temperature",        0.1},
            {"timeout",                 300}
    };
    for (auto &kv : default_config) {
        if (config_map.find(kv.first) == config_map.end()) {
            config_map[kv.first] = kv.second;
        }
    }

    Trainer<TicTacToe, TicTacToeModel> trainer(config_map);
    auto result = trainer.simulate_and_train(
            model,
            baseline_model,
            nullptr,
            [](
                    TicTacToeModel &model,
                    const std::unordered_map<std::string, float> &config
            ) {
                std::vector<SelfPlayResult> self_plays;
                std::cout << "Running " << int(config.at("simulation_cycle_games"))
                          << " simulations" << std::endl;
                for (int i = 0; i < int(config.at("simulation_cycle_games")); ++i) {
                    std::cout << "Running simulation " << i << std::endl;
                    self_plays.emplace_back(
                            mcts_model_self_play<TicTacToe, TicTacToeModel>(
                                    TicTacToe(),
                                    model,
                                    model,
                                    int(config.at("mcts_iterations")),
                                    10,
                                    config.at("simulation_temperature"),
                                    config.at("mcts_exploration")
                            )
                    );
                }
                return self_plays;
            }
    );

    torch::save(model, "models/tictactoe_model.pt");
    return result;
}