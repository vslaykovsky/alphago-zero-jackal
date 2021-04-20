#include <gtest/gtest.h>
#include <numeric>

#include "../src/mcts/mcts.h"
#include "../src/tictactoe/tictactoe.h"

using namespace std;




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