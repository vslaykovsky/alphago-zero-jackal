#include <gtest/gtest.h>
#include <numeric>

#include "../src/mcts/mcts.h"
#include "../src/tictactoe/tictactoe.h"

using namespace std;



TEST(MCTS, TicTacToe) {
    srand(12);
    TicTacToe ttt;
    while (!ttt.get_possible_actions().empty()) {
        const vector<int> &actions = ttt.get_possible_actions();
        ttt = ttt.take_action(actions[rand() % actions.size()]);
    }
    ASSERT_EQ(ttt.get_reward()[0], 1);
}

TEST(MCTS, MCTSTicTacToe) {
    TicTacToe game;
    srand(123);
    auto result = mcts_search(game, [](const TicTacToe &state) {
        MCTSStateActionValue result;
        auto actions = state.get_possible_actions();
        for (int a : actions) {
            result.action_proba[a] = 1. / actions.size();
        }
        result.state_value = state.get_reward();
        return result;
    }, 10000, 1.);
    ASSERT_EQ(4, result.best_action());
}