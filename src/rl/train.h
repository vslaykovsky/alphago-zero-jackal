#pragma once

#include "self_play.h"
#include "../util/utils.h"
#include "../../third_party/tensorboard_logger/include/tensorboard_logger.h"
#include <experimental/filesystem>
#include <utility>

class SelfPlayDataset {
public:

    struct Example {
        torch::Tensor x;
        torch::Tensor action_proba;
        torch::Tensor state_value;
    };

    explicit SelfPlayDataset(const std::vector<SelfPlayResult> &self_plays, int batch_size, bool shuffle = true,
                             torch::Device device = torch::kCPU,
                             bool only_terminal = false) {
        std::vector<Example> items;
        for (auto &self_play: self_plays) {
            int from_i = 0;
            if (only_terminal) {
                from_i = (int) self_play.states.size() - 1;
            }
            for (int i = from_i; i < self_play.states.size(); i++) {
                items.push_back(Example{
                        self_play.states[i],
                        torch::tensor({self_play.state_action_values[i].best_action()}),
                        self_play.reward_to_tensor()
                });
            }

        }
        if (shuffle)
            std::shuffle(items.begin(), items.end(), randomizer());
        for (int batch_idx = 0; batch_idx < items.size(); batch_idx += batch_size) {
            std::vector<torch::Tensor> x;
            std::vector<torch::Tensor> action_proba;
            std::vector<torch::Tensor> state_value;
            for (int i = batch_idx; i < std::min((int) items.size(), batch_idx + batch_size); ++i) {
                x.push_back(items[i].x);
                action_proba.push_back(items[i].action_proba);
                state_value.push_back(items[i].state_value);
            }
            examples.push_back(Example{
                    torch::cat({&x[0], x.size()}).to(device),
                    torch::cat({&action_proba[0], action_proba.size()}).to(device),
                    torch::stack({&state_value[0], state_value.size()}).to(device)
            });
        }
    };

    // Override the get method to load custom data.
    Example get(std::size_t index) const {
        return examples[index];
    };


    // Override the size method to infer the size of the data set.
    int size() const {
        return examples.size();
    };

    std::vector<Example> examples;
};

template<class TGame>
class StateValueDataset : public torch::data::Dataset<StateValueDataset<TGame>> {
public:
    explicit StateValueDataset(int size) {
        std::cout << "Generating StateValueDataset: " << size << " samples" << std::endl;
        for (int i = 0; i < size; ++i) {
            TGame game = random_self_play<TGame>();
            states.push_back(game.get_state());
            MCTSStateValue reward = game.get_reward();
            state_values.push_back(
                    torch::from_blob(&reward[0], at::IntArrayRef({(int) reward.size()}), torch::kFloat).clone());
        }
    }

    torch::data::Example<> get(size_t index) override {
        return torch::data::Example<>(states[index], state_values[index]);
    }

    torch::optional<size_t> size() const override {
        return states.size();
    }

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> state_values;
};


template<class TGame, class TModel>
float evaluate(TModel model, StateValueDataset<TGame> &ds) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto data_loader = torch::data::make_data_loader(ds.map(torch::data::transforms::Stack<>()), 32);
    int total = 0;
    float loss = 0.;
    for (auto &batch : *data_loader) {
        auto y = model(batch.data);
        auto target = batch.target;
        total += target.size(0);
        loss = loss + torch::mse_loss(y.value, target, at::Reduction::Sum).item().toDouble();
    }
    return loss / (float) total;
}


template<class TGame, class TModel>
float
compare_models(TModel model1, TModel model2, int trials, double model1_temperature, double model2_temperature) {
    model1->eval();
    model2->eval();
    torch::NoGradGuard no_grad;

    MCTSStateValue result(2);
    result.resize(2);
    int total_losses = 0;
    for (int i = 0; i < trials; ++i) {
        MCTSStateValue one_play_result;
        if (i % 2 == 0) {
            one_play_result = model_self_play<TGame>(model1, model2, model1_temperature,
                                                     model2_temperature).self_play_reward;
            assert(one_play_result.size() == 2);
            if (one_play_result[0] == -1) {
                total_losses++;
            }
        } else {
            one_play_result = model_self_play<TGame>(model2, model1, model2_temperature,
                                                     model1_temperature).self_play_reward;
            assert(one_play_result.size() == 2);
            if (one_play_result[1] == -1) {
                total_losses++;
            }
        }
    }
    return (float) total_losses / float(trials);
}

TensorBoardLogger gen_logger();


template<class TGame, class TModel>
class Trainer {
public:
    TensorBoardLogger logger;

    std::unordered_map<std::string, float> config;
    torch::Device device;

    explicit Trainer(
            std::unordered_map<std::string, float> pconfig = {},
            torch::Device device = torch::kCPU
    ) :
            logger(gen_logger()),
            device(device),
            config(std::move(pconfig)) {
        static std::unordered_map<std::string, float> default_config = {
                {"train_learning_rate",     1e-3},
                {"train_l2_regularization", 1e-4},
                {"train_replay_buffer",     1024},
                {"train_epochs",            10},
                {"train_batch_size",        32},

                {"simulation_cycles",       10},
                {"simulation_cycle_games",  256},
                {"simulation_temperature",  1.},
                {"simulation_threads",      1},
                {"simulation_max_turns",    1000},

                {"mcts_iterations",         100},
                {"mcts_exploration",        1.},

                {"eval_size",               100},
                {"eval_temperature",        1.},

                {"timeout",                 300}
        };
        for (auto &kv: default_config) {
            if (config.find(kv.first) == config.end()) {
                config[kv.first] = kv.second;
            }
        }
        for (auto &kv: config) {
            if (default_config.find(kv.first) == default_config.end()) {
                throw std::runtime_error("Invalid option " + kv.first);
            }
        }
    }

    float train(TModel model,
                const SelfPlayDataset &ds,
                TModel baseline_model,
                StateValueDataset<TGame> *eval_set,
                int &step) {
        using namespace std;
        cout << "running " << int(config["train_epochs"]) << " training epochs" << endl;
        using namespace std;
        model->train();
        torch::optim::Adam optimizer(model->parameters(),
                                     torch::optim::AdamOptions(config["train_learning_rate"]).weight_decay(
                                             config["train_l2_regularization"]));
        time_t t;
        time(&t);
        float benchmark_loss = 1e5;
        for (int epoch = 0; epoch < int(config["train_epochs"]); ++epoch) {
            std::cout << "train_epoch: " << epoch << std::endl;
            float train_loss = 0;
            for (int i = 0; i < ds.size(); ++i, ++step) {
                optimizer.zero_grad();
                const auto &example = ds.get(i);
                const auto &output = model(example.x);
                auto policy_loss = torch::nll_loss(output.policy, example.action_proba);
                auto state_value_loss = torch::mse_loss(output.value, example.state_value);
                auto loss = policy_loss + state_value_loss;
                loss.backward();
                train_loss += loss.item().toDouble();
                optimizer.step();
                logger.add_scalar("loss/train", step, loss.item().toDouble());
                logger.add_scalar("state-value-loss/train", step, state_value_loss.item().toDouble());
                logger.add_scalar("policy-loss/train", step, policy_loss.item().toDouble());
            }
            train_loss /= (float) ds.size();
            cout << "epoch:" << epoch << " train_loss:" << train_loss << endl;
//            benchmark_loss = compare_models<TGame, TModel>(model, baseline_model, int(config["eval_size"]),
//                                                           config["eval_temperature"], 1.);
//            logger.add_scalar("benchmark-loss/eval", step, benchmark_loss);
            if (eval_set != nullptr) {
                logger.add_scalar("value-loss/eval", step, evaluate<TGame, TModel>(model, *eval_set));
            }
            logger.add_scalar("epoch/train", step, (float) epoch);
            time_t t1;
            time(&t1);
            if (t1 - t > (int) config["timeout"]) {
                break;
            }
        }
        return benchmark_loss;
    }

    template<class S>
    float simulate_and_train(
            TModel model,
            TModel random_model,
            StateValueDataset<TGame> *eval_set,
            S gen_self_plays
    ) {
        using namespace std;
        TModel baseline_model;
        int step = 0;
        time_t t;
        time(&t);
        float loss = 1e5;
        for (int rl_epoch = 0; rl_epoch < int(config["simulation_cycles"]); ++rl_epoch) {
            cout << "simulation_cycle: " << rl_epoch << endl;
            vector<SelfPlayResult> self_plays(gen_self_plays(model, config));
            SelfPlayDataset ds(
                    vector<SelfPlayResult>(
                            self_plays.end() - min((int) self_plays.size(), int(config["train_replay_buffer"])),
                            self_plays.end()),
                    int(config["train_batch_size"]),
                    true,
                    device
            );
            loss = train(model, ds, baseline_model, eval_set, step);
            logger.add_scalar("rl_epoch/train", step, (float) rl_epoch);
            time_t t1;
            time(&t1);
            if (t1 - t > (int) config["timeout"]) {
                break;
            }
        }
        return loss;
    }
};

