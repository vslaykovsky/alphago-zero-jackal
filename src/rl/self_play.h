#pragma  once

#include <vector>
#include <unordered_map>
#include <ATen/core/Tensor.h>
#include <ATen/Tensor.h>
#include <torch/data/datasets/tensor.h>
#include <torch/torch.h>
#include "../mcts/mcts.h"
#include "../tictactoe/tictactoe_model.h"
#include "play.h"
#include "../util/utils.h"

struct SelfPlayResult {
    std::vector<MCTSStateActionValue> state_action_values;
    std::vector<torch::Tensor> states;
    MCTSStateValue self_play_reward;

    void add_state(const torch::Tensor &state, const MCTSStateActionValue &action_value) {
        state_action_values.push_back(action_value);
        states.push_back(state);
    }

    torch::Tensor reward_to_tensor() const {
        assert(self_play_reward.size() == 2);
        return torch::tensor({self_play_reward.at(0), self_play_reward.at(1)});
    }
};


template<class TGame, class F>
SelfPlayResult
mcts_model_self_play(TGame game, F state_action_value_func, int mcts_steps, int max_turns, float temperature,
                     float exploration, std::atomic<int> *turns = nullptr, bool verbose = false) {
    torch::NoGradGuard no_grad;
    SelfPlayResult self_play_result;
    MCTSStateActionValue state_action_value;
    int turn = 0;
    while (turn < max_turns && !game.get_possible_actions().empty()) {
        state_action_value = mcts_search(
                game,
                state_action_value_func,
                mcts_steps,
                exploration
        );
        self_play_result.add_state(game.get_state(), state_action_value);
        int action = state_action_value.sample_action(temperature);
        game = game.take_action(action);
        if (verbose) {
            std::cout << game << std::endl;
        }
        turn++;
        if (turns) {
            (*turns)++;
        }
//        std::cout << "turn " << turn << std::endl;
    }
    self_play_result.add_state(game.get_state(),
                               state_action_value);  // reuse last state_action_value. might be suboptimal
    self_play_result.self_play_reward = game.get_reward();
    return self_play_result;
}

template<class TGame, class TModel>
SelfPlayResult
mcts_model_self_play(TGame game, TModel model1, TModel model2, int mcts_steps, int max_turns, float temperature,
                     float exploration, std::atomic<int> *turns = nullptr, torch::Device device=torch::kCPU, bool verbose = false) {
    model1->to(device);
    model2->to(device);
    return mcts_model_self_play(game, [&model1, &model2, &device](const TGame &game) {
        MCTSStateActionValue result;
        GameModelOutput gmo;
        model1->named_parameters();

        if (game.get_current_player_id() == 0) {
            gmo = model1(game.get_state().to(device));
        } else {
            gmo = model2(game.get_state().to(device));
        }
        return to_state_action_value(gmo, game);
    }, mcts_steps, max_turns, temperature, exploration, turns, verbose);
}

template<class TGame, class TModel>
SelfPlayResult model_self_play(TModel player1, TModel player2,
                               float model1_temperature, float model2_temperature,
                               bool verbose = false) {
    using namespace std;
    TGame game;
    std::vector<TModel> models = {player1, player2};
    std::vector<float> temperature = {model1_temperature, model2_temperature};
    int model_idx = 0;
    SelfPlayResult result;
    result.states.push_back(game.get_state());
    while (!game.get_possible_actions().empty()) {
        auto game_state = game.get_state();
        auto model = models[model_idx];
        auto output = model(game_state);
        auto state_action_value = to_state_action_value(output, game);
        int action = state_action_value.sample_action(temperature[model_idx]);
        game = game.take_action(action);
        result.states.push_back(game.get_state());
        if (verbose) {
            cout << game << endl;
        }
        model_idx = (model_idx + 1) % 2;
    }
    result.self_play_reward = game.get_reward();
    return result;
}


template<class TGame>
TGame random_self_play() {
    TGame game;
    while (true) {
        const std::vector<int> &actions = game.get_possible_actions();
        if (actions.empty()){
            break;
        }
        int action = actions[rand() % actions.size()];
        game = game.take_action(action);
    }
    return game;
}