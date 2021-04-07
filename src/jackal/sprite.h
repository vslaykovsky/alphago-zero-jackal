#pragma once
#include <opencv2/opencv.hpp>
#include <string>


enum SpriteType {
    CANNIBAL = 0,
    EMPTY1 = 1,
    EMPTY2 = 2,
    EMPTY3 = 3,
    EMPTY4 = 4,
    SLIDE1 = 5,
    ARROW_R = 6,
    ARROW_RU = 7,
    ARROW_R_L = 8,
    ARROW_LB_RU = 9,
    ARROW_LU_R_B = 10,
    ARROW_L_R_U_B = 11,
    ARROW_LU_RU_LB_RB = 12,
    TRAP = 13,
    LABYRINTH2 = 14,
    LABYRINTH3 = 15,
    LABYRINTH4 = 16,
    LABYRINTH5 = 17,
    RUM = 18,
    CROCODILE = 19,
    BALLOON = 20,
    CANNON = 21,
    GOLD1 = 22,
    GOLD2 = 23,
    GOLD3 = 24,
    GOLD4 = 25,
    GOLD5 = 26,
    HORSE = 27,
    FORTRESS = 28,
    RESURRECTION = 29,
    PLANE = 30,
    SHIP1 = 32,
    SHIP2 = 33,
    SHIP3 = 34,
    SHIP4 = 35,
    CLOSED = 100,
};

class Sprite {
    cv::Mat image;
public:
    Sprite() = default;
    Sprite(const std::string& file_name, int y, int x);
    cv::Mat get_image(int size=318);

    static Sprite load(SpriteType sprite_type);
};