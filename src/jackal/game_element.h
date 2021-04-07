#pragma once

#include <opencv2/opencv.hpp>
#include "../util/utils.h"
#include "sprite.h"

class GameElement {
public:
    explicit GameElement(bool render = false, bool debug = false) : render(render), debug(debug) {

    }

    GameElement(const GameElement &ge) : state(ge.state.clone()), render(ge.render), debug(ge.debug) {

    }

    explicit GameElement(torch::Tensor t, bool render = false, bool debug = false) :
            state(std::move(t)),
            render(render),
            debug(debug) {

    }

    void render_tile(cv::Mat &image, int x, int y, SpriteType type, int rotation = 0);

    inline int width() const {
        return int(state.size(2));
    }

    inline int height() const {
        return int(state.size(1));
    }

    static const std::vector<cv::Point> &all_directions(bool diag = true);

    torch::Tensor state;
    bool render;
    bool debug;
};
