#pragma once

#include "../mcts/mcts.h"
#include "model.h"


MCTSActionValue filter_renormalize_actions(torch::Tensor tensor, const std::vector<int> &actions);

template<class T>
MCTSStateActionValue to_state_action_value(GameModelOutput &output, const T &game_state) {
    bool policy_enabled = output.policy.numel() > 0;
    assert(output.value.dim() == 2);
    auto value = output.value.to(torch::kCPU)[0];

    MCTSActionValue action_proba;
    if (policy_enabled) {
        assert(output.policy.dim() == 2);
        auto policy = output.policy.to(torch::kCPU)[0];
        auto actions = game_state.get_possible_actions();
        action_proba = filter_renormalize_actions(policy, actions);
    }
    return MCTSStateActionValue {
            MCTSStateValue(value.data_ptr<float>(), value.data_ptr<float>() + value.numel()),
            action_proba
    };
}
