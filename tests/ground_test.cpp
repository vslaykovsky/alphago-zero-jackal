#include <gtest/gtest.h>
#include "../src/jackal/jackal.h"
#include "../src/jackal/ground.h"
#include "../src/util/utils.h"

#include <sstream>
#include <string>

using namespace std;


TEST(GroundTest, TestGround) {
    srand(0);
    Ground g(3, 4, true);
    g.set_arrow(1, 1, SpriteType::ARROW_LU_R_B);
    g.set_gold(2, 1, 2);

    ASSERT_EQ(
            "(1,.,.) = \n  0  0  0  0\n  0  1  1  0\n  0  0  0  0\n\n"
            "(2,.,.) = \n  1  1  1  1\n  1  0  1  1\n  1  1  1  1\n\n"
            "(3,.,.) = \n  0  0  0  0\n  0  0  2  0\n  0  0  0  0\n\n"
            "(4,.,.) = \n  0  0  0  0\n  0  1  0  1\n  0  1  0  0\n\n"
            "(5,.,.) = \n  0  0  0  0\n  1  0  0  1\n  1  0  0  1\n\n"
            "(6,.,.) = \n  0  0  0  0\n  1  0  0  0\n  0  0  1  0\n\n"
            "(7,.,.) = \n  0  1  1  1\n  0  0  1  0\n  0  1  1  1\n\n"
            "(8,.,.) = \n  1  1  1  0\n  0  1  0  0\n  1  1  1  0\n\n"
            "(9,.,.) = \n  0  1  0  0\n  0  0  0  1\n  0  0  0  0\n\n"
            "(10,.,.) = \n  1  0  0  1\n  1  1  0  1\n  0  0  0  0\n\n"
            "(11,.,.) = \n  0  0  1  0\n  1  0  0  0\n  0  0  0  0\n[ CPUCharType{11,3,4} ]",
            to_string(g.state)
    );
}

TEST(GroundTest, TestGroundRender) {
    Ground g(7, 7, true, true);
    cv::imwrite("tmp/ground.png", g.image);
}

TEST(GroundTest, TestEncodeArrow) {
    auto t = Ground::encode_arrow(SpriteType::ARROW_R, 3);
    ASSERT_EQ(" 0\n 1\n 0\n 0\n 0\n 0\n 0\n 0\n[ CPUFloatType{8} ]", to_string(t));
    t = Ground::encode_arrow(SpriteType::ARROW_R, 0);
    ASSERT_EQ(" 0\n 0\n 0\n 0\n 1\n 0\n 0\n 0\n[ CPUFloatType{8} ]", to_string(t));
    t = Ground::encode_arrow(SpriteType::ARROW_R, 1);
    ASSERT_EQ(" 0\n 0\n 0\n 0\n 0\n 0\n 1\n 0\n[ CPUFloatType{8} ]", to_string(t));

    t = Ground::encode_arrow(SpriteType::ARROW_LU_R_B, 1);
    ASSERT_EQ(" 0\n 0\n 1\n 1\n 0\n 0\n 1\n 0\n[ CPUFloatType{8} ]", to_string(t));
}
