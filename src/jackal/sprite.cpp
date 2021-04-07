#include <unordered_map>
#include "sprite.h"
#include <experimental/filesystem>

using namespace std;


const int TILE_SIZE = 318;


Sprite::Sprite(const string &file_name, int y, int x) {
    auto fname = "img/" + file_name;
    auto curpath = std::experimental::filesystem::current_path();
    if (!std::experimental::filesystem::exists(fname)) {
        throw std::runtime_error("File is not found " + fname);
    }
    image = cv::imread(fname, 1);
    cv::Rect myROI(x * TILE_SIZE, y * TILE_SIZE, std::min(TILE_SIZE, image.cols - x * TILE_SIZE),
                   std::min(TILE_SIZE, image.rows - y * TILE_SIZE));
    image = image(myROI);
}

Sprite Sprite::load(SpriteType sprite_type) {
    static unordered_map<SpriteType, Sprite> SPRITES = {
            {SpriteType::EMPTY1,            Sprite("tiles1.png", 0, 0)},
            {SpriteType::EMPTY2,            Sprite("tiles1.png", 1, 1)},
            {SpriteType::EMPTY3,            Sprite("tiles1.png", 2, 2)},
            {SpriteType::EMPTY4,            Sprite("tiles1.png", 3, 0)},
            {SpriteType::SLIDE1,            Sprite("tiles1.png", 4, 2)},
            {SpriteType::ARROW_R,           Sprite("tiles2.png", 0, 0)},
            {SpriteType::ARROW_RU,          Sprite("tiles2.png", 0, 3)},
            {SpriteType::ARROW_R_L,         Sprite("tiles2.png", 1, 2)},
            {SpriteType::ARROW_LB_RU,       Sprite("tiles2.png", 2, 1)},
            {SpriteType::ARROW_LU_R_B,      Sprite("tiles2.png", 3, 0)},
            {SpriteType::ARROW_L_R_U_B,     Sprite("tiles2.png", 3, 3)},
            {SpriteType::ARROW_LU_RU_LB_RB, Sprite("tiles2.png", 4, 3)},
            {SpriteType::TRAP,              Sprite("tiles2.png", 5, 1)},
            {SpriteType::LABYRINTH2,        Sprite("tiles3.png", 0, 0)},
            {SpriteType::LABYRINTH3,        Sprite("tiles3.png", 1, 1)},
            {SpriteType::LABYRINTH4,        Sprite("tiles3.png", 2, 2)},
            {SpriteType::LABYRINTH5,        Sprite("tiles3.png", 2, 3)},
            {SpriteType::RUM,               Sprite("tiles3.png", 3, 0)},
            {SpriteType::CROCODILE,         Sprite("tiles3.png", 4, 0)},
            {SpriteType::BALLOON,           Sprite("tiles3.png", 5, 0)},
            {SpriteType::CANNON,            Sprite("tiles3.png", 5, 2)},
            {SpriteType::GOLD1,             Sprite("tiles4.png", 0, 0)},
            {SpriteType::GOLD2,             Sprite("tiles4.png", 1, 1)},
            {SpriteType::GOLD3,             Sprite("tiles4.png", 2, 2)},
            {SpriteType::GOLD4,             Sprite("tiles4.png", 3, 2)},
            {SpriteType::GOLD5,             Sprite("tiles4.png", 3, 3)},
            {SpriteType::HORSE,             Sprite("tiles4.png", 4, 0)},
            {SpriteType::FORTRESS,          Sprite("tiles4.png", 4, 2)},
            {SpriteType::RESURRECTION,      Sprite("tiles4.png", 5, 0)},
            {SpriteType::PLANE,             Sprite("tiles4.png", 5, 1)},
            {SpriteType::CANNIBAL,          Sprite("tiles4.png", 5, 2)},
            {SpriteType::SHIP1,             Sprite("ships.png", 0, 0)},
            {SpriteType::SHIP2,             Sprite("ships.png", 0, 1)},
            {SpriteType::SHIP3,             Sprite("ships.png", 0, 2)},
            {SpriteType::SHIP4,             Sprite("ships.png", 0, 3)},
            {SpriteType::CLOSED,            Sprite("closed_tile.png", 0, 0)},
    };
    return SPRITES[sprite_type];
}

cv::Mat Sprite::get_image(int size) {
    cv::Mat img;
    cv::resize(image, img, {size, size});
    return img;
}

