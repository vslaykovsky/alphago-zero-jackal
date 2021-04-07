#include "play.h"


MCTSActionValue filter_renormalize_actions(torch::Tensor tensor, const std::vector<int> &actions) {
    MCTSActionValue result;
    if (actions.empty()) {
        return result;
    }
    auto actions_tensor = torch::from_blob((int *) &actions[0], at::IntArrayRef({(int) actions.size()}),
                                           torch::kInt).clone().to(torch::kInt64);
    auto proba_tensor = tensor.softmax(0).index({actions_tensor}).contiguous();
    proba_tensor /= std::max(proba_tensor.sum().item<float>(), (float) 1e-8);
    std::vector<float> proba(proba_tensor.data_ptr<float>(), proba_tensor.data_ptr<float>() + proba_tensor.numel());
    for (int i = 0; i < actions.size(); ++i) {
        result[actions[i]] = proba[i];
        assert(proba[i] == proba[i]);
    }
    return result;
}
