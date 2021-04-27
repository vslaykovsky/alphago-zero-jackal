#pragma once

#include <torch/torch.h>
#include "../rl/model.h"

struct ConvModelImpl : torch::nn::Module {
    int input_channels;
    int output_channels;

    torch::nn::Conv2d conv2d{nullptr};
    torch::nn::BatchNorm2d batch_norm{nullptr};

    ConvModelImpl(int input_channels, int output_channels);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ConvModel);


struct ResModelImpl : torch::nn::Module {
    int channels;
    ConvModel conv{nullptr};
    torch::nn::Conv2d conv2d{nullptr};
    torch::nn::BatchNorm2d batch_norm{nullptr};

    explicit ResModelImpl(int channels);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResModel);


struct ValueHeadImpl : torch::nn::Module {
    int input_channels;
    int width;
    int height;
    int head_channels;
    torch::nn::Conv2d conv2d{nullptr};
    torch::nn::BatchNorm2d batch_norm{nullptr};
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear latent_state{nullptr};

    ValueHeadImpl(c10::IntArrayRef input_shape, int players, int head_features = 256, int head_channels = 2);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ValueHead);


struct PolicyHeadImpl : torch::nn::Module {
    int head_channels;
    torch::nn::Conv2d conv2d{nullptr};
    torch::nn::BatchNorm2d batch_norm{nullptr};
    torch::nn::Linear linear{nullptr};

    PolicyHeadImpl(c10::IntArrayRef input_shape, int head_channels);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PolicyHead);


struct JackalModelImpl : torch::nn::Module {
    ConvModel conv{nullptr};
    ValueHead value_head{nullptr};
    PolicyHead policy_head{nullptr};
    int blocks;

    explicit JackalModelImpl(c10::IntArrayRef input_shape = {1, 19, 12, 12},
                    int res_channels = 128,
                    int blocks = 10,
                    int players = 2,
                    bool action_value=true);

    GameModelOutput forward(torch::Tensor x);
};

TORCH_MODULE(JackalModel);


