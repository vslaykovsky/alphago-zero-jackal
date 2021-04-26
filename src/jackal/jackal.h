#pragma once

#include "sprite.h"
#include "ground.h"
#include "../util/utils.h"
#include "player.h"
#include <boost/functional/hash.hpp>
#include "action.h"
#include "../mcts/mcts.h"

#include <iostream>
#include <torch/torch.h>
#include <random>
#include <sstream>

using namespace torch::indexing;


class Jackal {
public:
    std::vector<Player> players;
    Ground ground;
    int current_player;
    int turn;
    bool render;
    bool debug;
    std::default_random_engine generator{123};


    Jackal(int height = 12, int width = 12, int players_num = 2, bool render = false, bool debug = false);

    int encode_action(const Action &action) const;

    int get_current_player_id() const;

    Jackal take_action(const torch::Tensor &action, float temperature = 1.);

    Jackal take_action(int action) const;

    Jackal take_action(const Action &action) const;

    std::vector<int> get_possible_actions() const;

    void store(const std::string &file_name) const;

    cv::Mat get_image(MCTSStateActionValue* mcts = nullptr);

    void load(torch::Tensor &state);

    void load(const std::string &file_name);

    inline int width() const {
        return ground.width();
    }

    inline int height() const {
        return ground.height();
    }

    torch::Tensor get_state() const;

    void set_next_player();

    int get_random_action() const;

    torch::Tensor encode_possible_actions() const;

    bool is_terminal();

    MCTSStateValue get_reward();

    Action decode_action(int action) const;
};

std::ostream &operator<<(std::ostream &os, const Jackal &j);