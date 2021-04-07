#pragma once

#include "action.h"
#include "game_element.h"
#include "ground.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


enum PlayerPlanes {
    PLANE_SHIP = 0,
    PLANE_PIRATES = 1,
    PLANE_SCORE = 2,
    PLANE_CURRENT_PLAYER = 3,
    PLAYER_PLANES_NUMBER = PLANE_CURRENT_PLAYER + 1
};

class Player : public GameElement {
public:
    int player_idx;

    explicit Player(int player_idx);

    Player(int player_idx, int w, int h, bool render = false, bool debug = false);

    void load(const torch::Tensor &tensor);

    void set_current_player(bool current_player);

    bool is_current_player() const;

    void inc_score(int score = 1);

    cv::Mat get_image();

    int get_score() const;

    Coords get_ship_coords() const ;

    std::vector<Coords> get_pirate_coords() const;

    void move_ship(const Coords &to);

    void remove_pirate(const Coords &from, bool all = false);

    void move_pirate(const Coords &from, const Coords &to, bool all = false);

    std::unordered_set<Action> get_pirate_actions(const Coords &coords, const Ground &ground) const;

    std::unordered_set<Action> get_ship_actions() const;

    std::vector<Action> get_possible_actions(const Ground &ground) const;

    int get_pirates(const Coords& p) const;
};
