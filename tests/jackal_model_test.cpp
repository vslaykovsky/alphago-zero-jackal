#include <gtest/gtest.h>
#include "../src/jackal/jackal.h"
#include "../src/jackal/game_model.h"


using namespace std;

TEST(GameModel, TestActionFilters) {
    Jackal game(7, 7, 2);
    JackalModel model(game.get_state().sizes(), 128, 10, 2);
    auto device = torch::kCUDA;
    model->to(device);
    auto state = game.get_state().to(device);
    vector<torch::Tensor> query = {game.encode_possible_actions()};
    torch::Tensor value;
    vector<torch::Tensor> policy;
    GameModelOutput output = model(state);

    ASSERT_EQ(2, output.value.dim());
    ASSERT_EQ(1, output.value.size(0));
    ASSERT_EQ(2, output.value.size(1)); // 2 players

    ASSERT_EQ(2, output.policy.dim());
    ASSERT_EQ(7 * 7 * 7 * 7 * 2, output.policy[0].size(0)); // 2 moves
}