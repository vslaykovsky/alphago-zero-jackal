#pragma once

#include <vector>
#include <unordered_map>
#include <ATen/core/Tensor.h>
#include <ATen/Tensor.h>
#include <torch/data/datasets/tensor.h>
#include <torch/torch.h>
#include "../mcts/mcts.h"

struct GameModelOutput {
    torch::Tensor policy;
    torch::Tensor value;
};

