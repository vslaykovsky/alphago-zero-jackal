#pragma once

#include "../rl/self_play.h"
#include "../rl/train.h"

#include <memory>
#include <utility>
#include "../../third_party/queue/concurrentqueue.h"
#include "../../third_party/queue/lightweightsemaphore.h"

#include "jackal.h"
#include "game_model.h"
#include <filesystem>

using namespace moodycamel;

struct TModelJob {
    torch::Tensor *state{nullptr};
    GameModelOutput *output{nullptr};
    LightweightSemaphore *semaphore{nullptr};
};

struct TTaskJob {
    TTaskJob(Jackal game,
             JackalModel model,
             const std::unordered_map<std::string, float> &config,
             SelfPlayResult &spr)
            : jackal(std::move(game)),
              model(std::move(model)),
              config(config),
              self_play_result(spr) {
    }

    Jackal jackal;
    JackalModel model;
    const std::unordered_map<std::string, float> &config;
    SelfPlayResult &self_play_result;
};

typedef ConcurrentQueue<std::unique_ptr<TTaskJob>> TTaskQueue;
typedef ConcurrentQueue<TModelJob> TModelQueue;


void
self_play_thread(int thread_num, TTaskQueue *task_queue, TModelQueue *model_queue, std::atomic<int> *jobs_completed,
                 std::atomic<int> *turns, std::atomic<bool> *terminated, TensorBoardLogger *logger) {
    using namespace std;
    LightweightSemaphore semaphore;
    std::unique_ptr<TTaskJob> task;
    cout << "[thread:" << thread_num << "] started thread" << endl;
    while (!*terminated) {
        while (!task_queue->try_dequeue(task)) {
            if (*terminated) {
                goto exit_thread;
            }
        }
        cout << "[thread:" << thread_num << "] started task" << endl;
        auto &config(task->config);
        task->self_play_result = mcts_model_self_play<>(
                task->jackal,
                [thread_num, model_queue, &semaphore](const Jackal &state) {
                    auto x = state.get_state();
                    GameModelOutput output;
                    TModelJob item{&x, &output, &semaphore};
                    model_queue->enqueue(item);
                    semaphore.wait();
                    return to_state_action_value(output, state);
                },
                int(config.at("mcts_iterations")),
                int(config.at("simulation_max_turns")),
                config.at("simulation_temperature"),
                config.at("mcts_exploration"),
                turns,
                logger
        );
        (*jobs_completed)++;
        cout << "[thread:" << thread_num << "] finished task" << endl;
    }
    exit_thread:
    cout << "[thread:" << thread_num << "] exit thread" << endl;
}

struct RequestContext {
    std::vector<TModelJob> items;
    torch::Tensor batch;
    GameModelOutput model_output;

    bool empty() const {
        return items.empty();
    }
};


bool read_request(TModelQueue &queue, RequestContext &request, std::atomic<bool> *terminated) {
    auto &items = request.items;
    items.clear();
    TModelJob item{};
    while (items.empty() && !*terminated) {
        while (queue.try_dequeue(item) && !*terminated) {
            items.push_back(item);
        }
    }
    if (!items.empty()) {
        std::vector<torch::Tensor> states;
        for (auto &i : items) {
            states.push_back(*i.state);
        }
        request.batch = torch::cat({&states[0], states.size()}).to(torch::kCUDA);
    }
    return !items.empty();
}

void reply(RequestContext &request) {
    auto &items(request.items);
    auto &model_output(request.model_output);
    model_output.policy = model_output.policy.to(torch::kCPU);
    model_output.value = model_output.value.to(torch::kCPU);
    for (int i = 0; i < items.size(); ++i) {
        items[i].output->policy = model_output.policy.index({i, "..."}).unsqueeze(0);
        items[i].output->value = model_output.value.index({i, "..."}).unsqueeze(0);
        items[i].semaphore->signal();
    }
}


void model_loop(JackalModel model, TModelQueue *queue, std::atomic<bool> *terminated, int *total_requests) {
    time_t tm;
    time(&tm);
    RequestContext cur_request;
    torch::NoGradGuard no_grad;
    model->eval();
    model->to(torch::kCUDA);
    while (!*terminated) {
        TModelJob item;
        if (read_request(*queue, cur_request, terminated)) {
            cur_request.model_output = (model)(cur_request.batch);
            reply(cur_request);
            (*total_requests) += cur_request.items.size();
        }
    }
}

void multithreaded_self_plays(const std::string &dir, int width, int height, JackalModel &model,
                              const std::unordered_map<std::string, float> &config, int players) {
    using namespace std;
    TTaskQueue task_queue;
    TModelQueue model_queue;

    std::vector<SelfPlayResult> self_plays;
    self_plays.resize(int(config.at("simulation_cycle_games")));
    std::cout << "Running " << self_plays.size() << " simulations" << std::endl;
    for (auto &self_play : self_plays) {
        task_queue.enqueue(
                std::make_unique<TTaskJob>(Jackal(height, width, players, config.at("simulation_render") > 0), model,
                                           config, self_play));
    }
    std::atomic<bool> terminated(false);
    std::atomic<int> jobs_completed(0);
    std::atomic<int> turns(0);
    int num_threads = int(config.at("simulation_threads"));
    std::vector<std::thread> sim_threads;
    sim_threads.reserve(num_threads);
    auto logger = gen_logger();
    for (int i = 0; i < num_threads; ++i) {
        sim_threads.emplace_back(
                std::thread(self_play_thread, i, &task_queue, &model_queue, &jobs_completed, &turns, &terminated,
                            self_plays.size() > 1 ? nullptr : &logger));
    }
    int total_requests = 0;
    std::thread model_thread(model_loop, model, &model_queue, &terminated, &total_requests);
    int prev_requests = 0;
    int jobs_persisted = 0;
    while (jobs_completed < self_plays.size()) {
        sleep(1);
        cout << "Simulations completed: " << jobs_completed << ". Total turns:" << turns << ". Total requests served: "
             << total_requests << ". Requests per second: " << (total_requests - prev_requests) << endl;
        if (jobs_completed - jobs_persisted >= 1000) {
            vector<SelfPlayResult> tmp_results;
            for (auto &self_play : self_plays) {
                if (!self_play.states.empty()) {
                    tmp_results.push_back(std::move(self_play));
                    jobs_persisted++;
                }
            }
            SelfPlayDataset ds(tmp_results, (int) config.at("train_batch_size"));
            ds.save_to_dir(dir);
        }
        prev_requests = total_requests;
    }
    terminated = true;
    for (auto &t: sim_threads) {
        t.join();
    }
    model_thread.join();
}


float
jackal_train(const std::string &dir, const std::unordered_map<std::string, float> &config_map, int width = 7,
             int height = 7, int players = 2) {
    using namespace std;
    auto device = torch::kCUDA;
    Jackal game(height, width, players);
    const at::Tensor &game_state = game.get_state().squeeze(0);
    c10::IntArrayRef dim = game_state.sizes();
    int channels = dim[0];
    JackalModel model(c10::IntArrayRef{1, channels, height, width});
    model->to(device);
    JackalModel baseline_model(c10::IntArrayRef{1, channels, height, width});
    baseline_model->to(device);

    std::unordered_map<std::string, float> config(config_map);
    // TODO tune up hyperparams
    std::unordered_map<std::string, float> default_config{
            {"train_learning_rate",         1e-4},
            {"train_l2_regularization",     0},
            {"train_replay_buffer",         1 >> 16},
            {"train_epochs",                1},
            {"train_batch_size",            128},

            {"simulation_cycle_games",      5000},
            {"simulation_cycles",           1000},
            {"simulation_temperature",      0.5},
            {"simulation_threads",          64},
            {"simulation_max_turns",        1000},

            {"mcts_iterations_first_cycle", 1},
            {"mcts_iterations",             256},
            {"mcts_exploration",            2},

            {"eval_size",                   0},
            {"eval_temperature",            0.1},

            {"timeout",                     600}
    };
    for (auto &kv : default_config) {
        if (config.find(kv.first) == config.end()) {
            config[kv.first] = kv.second;
        }
    }
    Trainer<Jackal, JackalModel> trainer(config, torch::kCUDA);
    auto result = trainer.simulate_and_train(
            dir,
            model,
            baseline_model,
            nullptr,
            [height, width, players](
                    const std::string& dir,
                    JackalModel &model,
                    const std::unordered_map<std::string, float> &config
            ) {
                return multithreaded_self_plays(dir, width, height, model, config, players);
            }
    );

    torch::save(model, "models/jackal_model.pt");
    return result;
}
