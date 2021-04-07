#include "game_model.h"

using namespace torch::indexing;

ConvModelImpl::ConvModelImpl(int input_channels, int output_channels) :
        input_channels(input_channels), output_channels(output_channels) {
    conv2d = register_module(
            "conv2d", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(input_channels, output_channels, 3).padding(1)
            ));
    batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(output_channels));
}

torch::Tensor ConvModelImpl::forward(torch::Tensor x) {
    x = conv2d(x);
    x = batch_norm(x);
    x = torch::relu(x);
    return x;
}

ResModelImpl::ResModelImpl(int channels) : channels(channels) {
    conv = register_module("conv", ConvModel(channels, channels));
    conv2d = register_module("conv2d", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1)

    ));
    batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(channels));

}

torch::Tensor ResModelImpl::forward(torch::Tensor x) {
    auto saved_x = x;
    x = conv(x);
    x = conv2d(x);
    x = batch_norm(x);
    x += saved_x;
    x = torch::relu(x);
    return x;
}

ValueHeadImpl::ValueHeadImpl(c10::IntArrayRef input_shape, int players) {
    input_channels = (int) input_shape[0];
    height = (int) input_shape[1];
    width = (int) input_shape[2];
    head_channels = 2;
    conv2d = register_module("conv2d", torch::nn::Conv2d(input_channels, head_channels, 1));
    batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(head_channels));
    int head_features = 256;
    linear1 = register_module("linear1", torch::nn::Linear(head_channels * width * height, head_features));
    linear2 = register_module("latent_state", torch::nn::Linear(head_features, players));
}

torch::Tensor ValueHeadImpl::forward(torch::Tensor x) {
    x = conv2d(x);
    x = batch_norm(x);
    x = torch::relu(x);
    x = torch::reshape(x, {x.size(0), -1});
    x = linear1(x);
    x = torch::relu(x);
    x = linear2(x);
    x = torch::tanh(x);
    return x;
}

PolicyHeadImpl::PolicyHeadImpl(c10::IntArrayRef input_shape, int head_channels) : head_channels(head_channels) {
    int input_channels = (int) input_shape[0];
    int height = (int) input_shape[1];
    int width = (int) input_shape[2];
    conv2d = register_module("conv2d", torch::nn::Conv2d(input_channels, head_channels, 1));
    batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(head_channels));
    linear = register_module("linear", torch::nn::Linear(head_channels * width * height,
                                                         (width * height) * (width * height) * 2));
}

torch::Tensor PolicyHeadImpl::forward(torch::Tensor x) {
    x = conv2d(x);
    x = batch_norm(x);
    x = torch::relu(x);
    x = torch::reshape(x, {x.size(0), -1});
    x = linear(x);
    return torch::log_softmax(x, 1);
}

JackalModelImpl::JackalModelImpl(c10::IntArrayRef input_shape, int res_channels, int blocks, int players) : blocks(blocks) {
    int input_channels = (int) input_shape[0];
    int height = (int) input_shape[1];
    int width = (int) input_shape[2];
    conv = register_module("conv", ConvModel(input_channels, res_channels));
    for (int i = 0; i < blocks; ++i) {
        register_module("res" + std::to_string(i), ResModel(res_channels));
    }
    policy_head = register_module("policy_head", PolicyHead(c10::IntArrayRef({res_channels, height, width}), 4));
    value_head = register_module("value_head", ValueHead(c10::IntArrayRef({res_channels, height, width}), players));
}

GameModelOutput JackalModelImpl::forward(torch::Tensor x) {
    x = conv(x);
    for (int i = 0; i < blocks; ++i) {
        auto res_model = named_children()["res" + std::to_string(i)]->as<ResModel>();
        x = res_model->forward(x);
    }
    auto policy = policy_head(x);
    auto value = value_head(x);
    return GameModelOutput{policy, value};
}


