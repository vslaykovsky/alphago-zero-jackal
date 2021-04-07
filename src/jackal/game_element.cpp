#include "game_element.h"

void GameElement::render_tile(cv::Mat &image, int x, int y, SpriteType type, int rotation) {
    if (render) {
        auto img = Sprite::load(type).get_image(TILE_SIZE);
        for (int i = 0; i < rotation; ++i) {
            cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
        }
        if (image.channels() == 4) {
            cv::cvtColor(img, img, cv::COLOR_RGB2RGBA);
        }
        img.copyTo(image.rowRange(y * TILE_SIZE, (y + 1) * TILE_SIZE).colRange(x * TILE_SIZE, (x + 1) * TILE_SIZE));
    }
}

const std::vector<cv::Point> &GameElement::all_directions(bool diag) {
    static const std::vector<cv::Point> directions8 = {{-1, -1},
                                                       {0,  -1},
                                                       {1,  -1},
                                                       {-1, 0},
                                                       {1,  0},
                                                       {-1, 1},
                                                       {0,  1},
                                                       {1,  1}};
    static const std::vector<cv::Point> directions4 = {
            {0,  -1},
            {-1, 0},
            {1,  0},
            {0,  1},
    };
    return diag ? directions8 : directions4;
}
