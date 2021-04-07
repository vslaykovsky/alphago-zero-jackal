#pragma once
#include <torch/torch.h>
#include "../rl/model.h"


class TicTacToeModelImpl : public torch::nn::Module {
public:
    TicTacToeModelImpl(int hidden=20) {
        linear1 = (register_module("linear1", torch::nn::Linear(10, hidden)));
        latent_state = (register_module("latent_state", torch::nn::Linear(hidden, hidden)));
        latent_action = (register_module("latent_action", torch::nn::Linear(hidden, hidden)));
        linear_action = (register_module("linear_action", torch::nn::Linear(hidden, 9)));
        linear_state = (register_module("linear_state", torch::nn::Linear(hidden, 2)));
    }

    GameModelOutput forward(torch::Tensor x) {
        auto x_linear1 = torch::relu(linear1(x));

        auto x_latent_state = torch::relu(latent_state(x_linear1));
        auto state = torch::tanh(linear_state(x_latent_state));

        auto x_latent_action = torch::relu(latent_action(x_linear1));
        auto action = torch::log_softmax(linear_action(x_latent_action), -1);

        return GameModelOutput{action, state};
    }


    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear latent_state{nullptr};
    torch::nn::Linear latent_action{nullptr};
    torch::nn::Linear linear_action{nullptr};
    torch::nn::Linear linear_state{nullptr};
};

TORCH_MODULE(TicTacToeModel);