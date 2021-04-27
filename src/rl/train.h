#pragma once

#include "self_play.h"
#include "../util/utils.h"
#include "../../third_party/tb_logger/include/tensorboard_logger.h"
#include <experimental/filesystem>
#include <utility>
#include <filesystem>


inline std::vector<std::string> get_selfplay_files(const std::string &dir) {
    std::vector<std::string> selfplays;
    for (auto &p: std::filesystem::directory_iterator(dir)) {
        std::string s = p.path();
        if (s.find("selfplay") != std::string::npos) {
            selfplays.push_back(s);
        }
    }
    return selfplays;
}

class SelfPlayDataset {
    torch::Device device;
public:

    struct Example {
        torch::Tensor x;
        torch::Tensor action_proba;
        torch::Tensor state_value;
    };


    explicit SelfPlayDataset(torch::Device device = torch::kCPU) : device(device) {}

    explicit SelfPlayDataset(const std::vector<SelfPlayResult> &self_plays, int batch_size, bool shuffle = true,
                             torch::Device device = torch::kCPU, float sampling=1.0) : device(device) {
        std::vector<Example> items;
        for (auto &self_play: self_plays) {
            int from_i = 0;
            for (int i = from_i; i < self_play.states.size(); i++) {
                items.push_back(Example{
                        self_play.states[i],
                        torch::tensor({self_play.state_action_values[i].best_action()}),
                        self_play.reward_to_tensor()
                });
            }

        }
        if (shuffle)
            std::shuffle(items.begin(), items.end(), get_generator());
        if (sampling < 1.0) {
            items.erase(items.begin() + int(items.size() * sampling), items.end());
        }
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

    void save_tensor(torch::Tensor &t, std::ostream &os) {
        std::ostringstream o(std::ios::out | std::ios::binary);
        torch::save(t, o);
        int32_t sz = o.str().size();
        os.write((char *) &sz, 4);
        os.write(o.str().c_str(), sz);
    }

    torch::Tensor load_tensor(std::istream &is) {
        int32_t sz;
        is.read((char *) &sz, 4);
        std::unique_ptr<char[]> buf(new char[sz]);
        is.read(buf.get(), sz);
        std::istringstream i(std::string(buf.get(), buf.get() + sz));
        torch::Tensor t;
        torch::load(t, i);
        return t;
    }


    void save_to_dir(const std::string &dir) {
        auto selfplay_files = get_selfplay_files(dir);
        const std::string &selfplay_file = dir + "/selfplay_" + std::to_string(selfplay_files.size()) + ".bin";
        save(selfplay_file);
    }

    void save(const std::string &fname) {
        std::ofstream f(fname, std::ios::out | std::ios::binary);
        int32_t size = examples.size();
        f.write((char *) &size, 4);
        for (auto &ex: examples) {
            save_tensor(ex.x, f);
            save_tensor(ex.action_proba, f);
            save_tensor(ex.state_value, f);
        }
    }

    void load(const std::string &fname) {
        examples.clear();
        std::ifstream f(fname, std::ios::in | std::ios::binary);
        int32_t size;
        f.read((char *) &size, 4);
        for (int i = 0; i < size; ++i) {
            Example ex;
            ex.x = load_tensor(f).to(device);
            ex.action_proba = load_tensor(f).to(device);
            ex.state_value = load_tensor(f).to(device);
            examples.push_back(ex);
        }
    }
};


template<class TModel>
float evaluate(TModel model, const std::string &dir, float sampling, torch::Device device = torch::kCPU) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto selfplay_files = get_selfplay_files(dir);
    float loss = 0.;
    int total = 0;
    for (auto selfplay_file : selfplay_files) {
        std::cout << "Eval file: " << selfplay_file << std::endl;
        SelfPlayDataset ds(device);
        ds.load(selfplay_file);
        for (int i = 0; i < ds.size() * sampling; ++i) {
            auto sample = ds.get(i);
            auto y = model(sample.x);
            auto target = sample.state_value;
            loss = loss + torch::mse_loss(y.value, target).item().toDouble();
            total++;
        }
    }
    return loss / total;

}


template<class TGame, class TModel>
float
compare_models(TModel model1, TModel model2, int trials, double model1_temperature, double model2_temperature) {
    if (trials == 0) {
        return 0.;
    }
    model1->eval();
    model1->to(torch::kCPU);
    model2->eval();
    model2->to(torch::kCPU);
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
                {"train_learning_rate",         1e-3},
                {"train_l2_regularization",     1e-4},
                {"train_replay_buffer",         1 >> 16},
                {"train_epochs",                10},
                {"train_batch_size",            32},

                {"simulation_cycles",           10},
                {"simulation_cycle_games",      256},
                {"simulation_temperature",      1.},
                {"simulation_threads",          1},
                {"simulation_max_turns",        1000},

                {"mcts_iterations",             100},
                {"mcts_iterations_first_cycle", 100},
                {"mcts_exploration",            1.},

                {"eval_size",                   100},
                {"eval_temperature",            1.},

                {"timeout",                     300}
        };
        for (auto &kv: default_config) {
            if (config.find(kv.first) == config.end()) {
                config[kv.first] = kv.second;
            }
        }
        std::cout << "Starting Trainer with configuration:" << std::endl;
        for (auto &kv: config) {
            std::cout << "  " << kv.first << ": \t" << kv.second << std::endl;
        }
        for (auto &kv: config) {
            if (default_config.find(kv.first) == default_config.end()) {
                std::cout << "Unknown option " << kv.first << std::endl;
            }
        }
    }

    float train(const std::string &dir,
                SelfPlayDataset *eval_set,
                int &step,
                int channels = 128,
                int blocks = 10,
                int players = 2) {
        TGame game((int) config["jackal_height"], (int) config["jackal_width"], (int) config["jackal_players"]);
        auto dims = game.get_state().sizes();

        TModel model(dims, channels, blocks, players, config["enable_action_value"] > 0);
        TModel baseline_model(dims, channels, blocks, players, config["enable_action_value"] > 0);

        auto model_path = dir + "/model.bin";
        if (std::filesystem::exists(model_path)) {
            torch::load(model, model_path);
            torch::load(baseline_model, model_path);
        }
        std::cout << model << std::endl;

        using namespace std;
        cout << "running " << int(config["train_epochs"]) << " training epochs" << endl;
        using namespace std;
        model->train();
        model->to(device);
        torch::optim::Adam optimizer(model->parameters(),
                                     torch::optim::AdamOptions(config["train_learning_rate"]).weight_decay(
                                             config["train_l2_regularization"]));
        time_t t;
        time(&t);
        float benchmark_loss = 1e5;

        logger.add_scalar("state-value-loss/eval", 0,
                          evaluate<TModel>(model,
                                           dir + "/eval",
                                           config["train_replay_sampling_rate"],
                                           device));

        for (int epoch = 0; epoch < int(config["train_epochs"]); ++epoch) {
            auto selfplay_files = get_selfplay_files(dir + "/train");
            for (int spf = 0; spf < selfplay_files.size(); spf++) {
                auto selfplay_file = selfplay_files[spf];
                std::cout << "train_epoch: " << epoch << " train file:" << selfplay_file << std::endl;
                SelfPlayDataset ds(device);
                ds.load(selfplay_files.back());

                float train_loss = 0;
                for (int i = 0; i < (int) (ds.size()); ++i, ++step) {
                    if (rand01() > config["train_replay_sampling_rate"])
                        continue;
                    optimizer.zero_grad();
                    const auto &example = ds.get(i);
                    const auto &output = model(example.x.to(device));
                    torch::Tensor loss;
                    auto state_value_loss = torch::mse_loss(output.value, example.state_value);
                    if (config.at("enable_action_value") > 0) {
                        auto policy_loss = torch::nll_loss(output.policy, example.action_proba);
                        loss = policy_loss + state_value_loss;
                        logger.add_scalar("policy-loss/train", step, policy_loss.item().toDouble());
                    } else {
                        loss = state_value_loss;
                    }
                    loss.backward();
                    train_loss += loss.item().toDouble();
                    optimizer.step();
                    logger.add_scalar("loss/train", step, loss.item().toDouble());
                    logger.add_scalar("state-value-loss/train", step, state_value_loss.item().toDouble());
                }
                train_loss /= (float) ds.size();
                cout << "epoch:" << epoch << " train_loss:" << train_loss << endl;
                benchmark_loss = compare_models<TGame, TModel>(model, baseline_model, int(config["eval_size"]),
                                                               config["eval_temperature"], 1.);
                logger.add_scalar("benchmark-loss/eval", step, benchmark_loss);
                if (spf % max(1, int(selfplay_files.size() / 10)) == 0) {
                    logger.add_scalar("state-value-loss/eval", step,
                                      evaluate<TModel>(model,
                                                       dir + "/eval",
                                                       config["train_replay_sampling_rate"],
                                                       device)
                    );
                }
                logger.add_scalar("epoch/train", step, (float) epoch);
                time_t t1;
                time(&t1);
                if (t1 - t > (int) config["timeout"]) {
                    break;
                }
            }
        }
        torch::save(model, model_path);
        return benchmark_loss;
    }

    template<class S>
    float simulate_and_train(
            const std::string &dir,
            TModel model,
            TModel random_model,
            SelfPlayDataset *eval_set,
            S gen_self_plays
    ) {
        using namespace std;
        TModel baseline_model;
        int step = 0;
        time_t t;
        time(&t);
        float loss = 1e5;
        auto mcts_iterations = config["mcts_iterations"];
        auto mcts_iterations_first_cycle = config["mcts_iterations_first_cycle"];
        for (int rl_epoch = 0; rl_epoch < int(config["simulation_cycles"]); ++rl_epoch) {
            cout << "simulation_cycle: " << rl_epoch << endl;
            if (rl_epoch == 0) {
                config["mcts_iterations"] = mcts_iterations_first_cycle;
            } else {
                config["mcts_iterations"] = mcts_iterations;
            }
            gen_self_plays(dir, model, config);
            SelfPlayDataset ds(device);
            ds.load(get_selfplay_files(dir).back()); // todo refactor train
            loss = train(dir, eval_set, step);
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

