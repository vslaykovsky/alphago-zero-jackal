#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include "../src/jackal/jackal.h"
#include "../src/jackal/ground.h"
#include "../src/util/utils.h"


using namespace std;


TEST(PlayerTest, RenderTest) {
    Player p(0, 5, 5, true, true);
    cv::imwrite("tmp/player.png", p.get_image());
}

TEST(PlayerTest, ShipPositionTest) {
    Player p(0, 5, 5, true, true);
    ASSERT_EQ(cv::Point(0, 2), p.get_ship_coords());

    Player p1(1, 5, 5, true, true);
    ASSERT_EQ(cv::Point(4, 2), p1.get_ship_coords());

    Player p2(2, 5, 5, true, true);
    ASSERT_EQ(cv::Point(2, 0), p2.get_ship_coords());
}

TEST(PlayerTest, ShipActionsTest) {
    Player p(0, 5, 5, true, true);
    ASSERT_EQ(cv::Point(0, 2), p.get_ship_coords());

    auto actions = p.get_ship_actions();
    ASSERT_EQ(2, actions.size());
    ASSERT_EQ(Action(Coords(0, 2), Coords(0, 3), true), *actions.begin());
    ASSERT_EQ(Action(Coords(0, 2), Coords(0, 1), true), *(++actions.begin()));
}