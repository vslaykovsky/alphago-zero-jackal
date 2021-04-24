#include <gtest/gtest.h>
#include "../src/jackal/jackal.h"
#include "../src/jackal/game_model.h"
#include "../src/rl/train.h"
#include <experimental/filesystem>
#include "helpers.h"

#include <sstream>
#include <string>

using namespace std;

TEST(JackalTest, JackalRenderer) {
    Jackal j(7, 7, 2, true, true);
    cv::imwrite("tmp/jackal.png", j.get_image());
}

TEST(JackalTest, RandomSelfPlayRender) {
    TestGuard g;

    Jackal jackal(7, 7, 2, true, true);
    std::experimental::filesystem::create_directories("tmp/rand_self_play");
    cv::imwrite("tmp/rand_self_play/game" + to_string(jackal.turn) + ".png", jackal.get_image());
    for (int i = 0; !jackal.is_terminal() && i < 10; ++i) {
        jackal = jackal.take_action(jackal.get_random_action());
        cv::imwrite("tmp/rand_self_play/game" + to_string(jackal.turn) + ".png", jackal.get_image());
    }
}

TEST(JackalTest, RandomSelfPlayPerformance) {
    TestGuard g;

    Jackal jackal(7, 7, 2, false, false);
    auto t1 = clock();
    while (jackal.turn < 10000 && !jackal.is_terminal()) {
        jackal = jackal.take_action(jackal.get_random_action());
    }
    auto t2 = clock();
    auto diff = float(t2 - t1) / CLOCKS_PER_SEC;
    cout << jackal.turn << " steps, time diff: " << diff << endl;
    ASSERT_LT(diff, 1);
}

TEST(JackalTest, StateUpdates) {
    Jackal jackal(7, 7, 2, false, false);
    Jackal j1(jackal.take_action(jackal.get_random_action()));
    ASSERT_TRUE(torch::all(torch::eq(j1.get_state(), j1.get_state())).item().toBool());
    ASSERT_FALSE(torch::all(torch::eq(j1.get_state(), jackal.get_state())).item().toBool());
}


TEST(JackalTest, FullTrainingCycle) {
    TestGuard g;


    unordered_map<string, float> config{
            {"train_learning_rate",     1e-4},
            {"train_l2_regularization", 1e-6},
            {"train_replay_buffer",     2048},
            {"train_epochs",            4},
            {"train_batch_size",        32},

            {"simulation_cycle_games",  2},
//            {"simulation_cycle_games",  64},
            {"simulation_cycles",       1000},
            {"simulation_temperature",  0.3},
            {"simulation_max_turns",    10},
//            {"simulation_max_turns",    1000},

            {"mcts_iterations",         256},
            {"mcts_exploration",        2},

//            {"eval_size",               200},
            {"eval_size",               0},
            {"eval_temperature",        0.1},
            {"timeout",                 60}
    };
    Jackal game;
    JackalModel model;
    JackalModel rnd_model;
    Trainer<Jackal, JackalModel> trainer(config);
    auto result = trainer.simulate_and_train(
            "tmp/test",
            model,
            rnd_model,
            nullptr,
            [](const std::string& dir, JackalModel &model, unordered_map<string, float> &config) {
                vector<SelfPlayResult> results;
                for (int i = 0; i < int(config["simulation_cycle_games"]); ++i) {
                    Jackal game;
                    std::cout << "Self play " << i << std::endl;
                    results.push_back(
                            mcts_model_self_play(
                                    game, model, model, int(config["mcts_iterations"]),
                                    int(config["simulation_max_turns"]), 1, 1, nullptr, torch::kCUDA
                            )
                    );
                }
                return results;
            });
    torch::save(model, "tmp/jackal_model.pt");
}

TEST(JackalTest, TestEncodeDecodeAction) {
    Jackal j;
    auto actions = j.get_possible_actions();
    for (auto a : actions) {
        ASSERT_EQ(a, j.encode_action(j.decode_action(a)));
    }
}
