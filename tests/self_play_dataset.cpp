#include <gtest/gtest.h>
#include "../src/tictactoe/tictactoe_model.h"
#include "../src/rl/self_play.h"
#include "../src/rl/train.h"
#include "../src/tictactoe/tictactoe.h"
#include "helpers.h"
#include <filesystem>

using namespace std;

TEST(SPDS, TestSelfPlayDataSet) {
    TestGuard g;
    TicTacToeModel model;
    TicTacToe game;
    auto self_play = mcts_model_self_play<TicTacToe, TicTacToeModel>(game, model, model, 1, 10, 1., 1.);
    auto ds = SelfPlayDataset(std::vector<SelfPlayResult>{self_play}, 1, false);
    ds.save("tmp/testds.bin");
    ds.load("tmp/testds.bin");
    auto ex = ds.examples.back();
    ASSERT_EQ(" 1 -1 -1  1  1  1 -1 -1  1  1\n[ CPUFloatType{1,10} ]",
              to_string(ex.x));
    ASSERT_EQ(" 0  0\n[ CPUFloatType{1,2} ]",
              to_string(ex.state_value));
    ASSERT_EQ(" 3\n[ CPULongType{1} ]",
              to_string(ex.action_proba));
}

TEST(SPDS, AnalyzeSPDS) {
    SelfPlayDataset ds;
    auto fnames = get_selfplay_files("tmp/jackal/epoch0/");
    int nonz = 0;
    int total = 0;
    auto state_values = torch::zeros({2});
    ds.load(fnames);

    for (int i = 0; i < ds.examples.size(); ++i) {
        auto ex = ds.examples[i];
        cout << "batch " << i << endl;
        cout << ex.action_proba << endl;
        cout << ex.state_value << endl;
        state_values += ex.state_value.sum(0);
        nonz += torch::count_nonzero(ex.state_value).item().toInt();
        total += ex.state_value.numel();

    }
    cout << "---" << endl;
    cout << float(nonz) / total << endl;
    cout << state_values / total << endl;
    cout << state_values << endl;
    cout << total << endl;
    total = 0;
    state_values = torch::zeros({2});
}