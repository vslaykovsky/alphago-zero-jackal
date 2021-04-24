#pragma once

#include "../mcts/mcts.h"

#include <memory>
#include <vector>
#include <torch/torch.h>

class TicTacToe {
public:
    int turn;
    std::vector<std::vector<int>> field;

    explicit TicTacToe(int size = 3);

    explicit TicTacToe(const std::vector<std::vector<int>> &field);

    int win() const;

    MCTSStateValue get_reward() const;

    std::vector<int> get_possible_actions() const;

    TicTacToe take_action(int action) const;

    int get_current_player_id() const;

    torch::Tensor get_state() const;

    cv::Mat get_image() const;
};

std::ostream &operator<<(std::ostream &os, const TicTacToe &ttt);