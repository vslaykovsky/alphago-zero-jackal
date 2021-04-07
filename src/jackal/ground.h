#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "../util/utils.h"
#include "game_element.h"


enum GroundPlanes {
    PLANE_GROUND = 0,
    PLANE_DELAYS = 1,
    PLANE_GOLD = 2,
    PLANE_DIRECTIONS = 3,
    GROUND_PLANES_NUMBER = PLANE_DIRECTIONS + 8
};


class Ground : public GameElement {
public:

    explicit Ground(bool render = false, bool debug = false);


    Ground(int height, int width, bool render = false, bool debug = false);

    void load(const torch::Tensor &tensor);

    void render_arrows(int y, int x);

    void set_gold(int x, int y, int coins);

    int get_gold(int x, int y) const;

    void set_default_directions(int x, int y);

    void set_arrow(int x, int y, SpriteType arrow_type, int rotation = 0);

    bool is_ground(int x, int y) const;

    bool is_arrow(int x, int y) const;

    int get_delay(int x, int y) const;

    std::vector<Coords> get_directions(int x, int y) const;

    static const std::vector<Coords> &get_arrow_directions(SpriteType st);

    static torch::Tensor encode_arrow(SpriteType arrow, int rotation);

    cv::Mat get_image();

    cv::Mat image;

    void move_gold(const Coords &from, const Coords &to);

    int total_gold();

    void remove_gold(Coords point);

    bool valid_coord(int x, int y) const;
};
