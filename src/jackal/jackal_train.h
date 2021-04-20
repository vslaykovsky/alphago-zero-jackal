#pragma once

#include "../rl/self_play.h"
#include "../rl/train.h"

#include <memory>
#include <utility>
#include "../../third_party/queue/concurrentqueue.h"
#include "../../third_party/queue/lightweightsemaphore.h"

#include "jackal.h"
#include "game_model.h"

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


void self_play_thread(TTaskQueue *task_queue, TModelQueue *model_queue, std::atomic<int> *jobs_comleted,
                      std::atomic<int> *turns, std::atomic<bool>* terminated) {
    LightweightSemaphore semaphore;
    std::unique_ptr<TTaskJob> task;
    while (!*terminated) {
        while (!task_queue->try_dequeue(task));
        auto &config(task->config);
        task->model->eval();
        task->self_play_result = mcts_model_self_play<>(
                task->jackal,
                [model_queue, &semaphore](const Jackal &state) {
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
                turns
        );
        (*jobs_comleted)++;
    }
}

struct RequestContext {
    std::vector<TModelJob> items;
    torch::Tensor batch;
    GameModelOutput model_output;
};


bool read_request(TModelQueue &queue, RequestContext &request) {
    auto &items = request.items;
    bool terminate = false;
    items.clear();
    TModelJob item{};
    while (items.empty()) {
        if (queue.try_dequeue(item)) {
            if (item.semaphore == nullptr) {
                terminate = true;
            } else {
                items.push_back(item);
            }
        }
    }
    std::vector<torch::Tensor> states;
    for (auto &i : items) {
        states.push_back(*i.state);
    }
    request.batch = torch::cat({&states[0], states.size()}).to(torch::kCUDA);
    return terminate;
}

void reply(RequestContext &request) {
    auto &items(request.items);
    auto &model_output(request.model_output);
    model_output.policy = model_output.policy.to(torch::kCPU);
//    model_output.value = model_output.value.to(torch::kCPU);
    for (int i = 0; i < items.size(); ++i) {
        items[i].output->policy = model_output.policy.index({i, "..."});
        items[i].semaphore->signal();
    }
}

// root only
void set_high_thread_priority() {
    int priority_max = sched_get_priority_max(SCHED_FIFO);
    int priority_min = sched_get_priority_min(SCHED_FIFO);
    pthread_t main_id = pthread_self();
    struct sched_param param{};
    param.sched_priority=priority_max;
    int status = pthread_setschedparam(main_id, SCHED_FIFO, &param);
    if (status != 0)
        perror("pthread_setschedparam");
}

void model_loop(JackalModel* model, TModelQueue* queue, std::atomic<bool>* terminated) {
//    set_high_thread_priority();
    time_t tm;
    time(&tm);
    RequestContext cur_request;
    while (!*terminated) {
        TModelJob item;
        read_request(*queue, cur_request);
        cur_request.model_output = (*model)(cur_request.batch);
        reply(cur_request);
    }
}

std::vector<SelfPlayResult> multithreaded_self_plays(
        int height,
        int width,
        int players,
        JackalModel &model,
        const std::unordered_map<std::string, float> &config
) {
    using namespace std;
    TTaskQueue task_queue;
    TModelQueue model_queue;

    std::vector<SelfPlayResult> self_plays;
    self_plays.resize(int(config.at("simulation_cycle_games")));
    std::cout << "Running " << self_plays.size() << " simulations" << std::endl;
    for (auto &self_play : self_plays) {
        task_queue.enqueue(
                std::make_unique<TTaskJob>(Jackal(height, width, players), model, config, self_play));
    }
    std::atomic<bool> terminated(false);
    std::atomic<int> jobs_completed(0);
    std::atomic<int> turns(0);
    int num_threads = int(config.at("simulation_threads"));
    std::vector<std::thread> sim_threads;
    for (int i = 0; i < num_threads; ++i) {
        sim_threads.emplace_back(std::thread(self_play_thread, &task_queue, &model_queue, &jobs_completed, &turns, &terminated));
    }
    std::thread model_thread(model_loop, &model, &model_queue, &terminated);
    while (jobs_completed < self_plays.size()) {
        sleep(1);
        cout << "Simulations completed: " << jobs_completed << ". Total turns:" << turns << endl;
    }
    terminated = true;
    for (auto &t: sim_threads) {
        t.join();
    }
    model_thread.join();
    return self_plays;
}


float
jackal_train(const std::unordered_map<std::string, float> &config_map, int width = 7, int height = 7, int players = 2) {
    using namespace std;
    auto device = torch::kCUDA;
    Jackal game(height, width, players);
    const at::Tensor &game_state = game.get_state().squeeze(0);
    c10::IntArrayRef dim = game_state.sizes();
    int channels = dim[0];
    JackalModel model(c10::IntArrayRef{channels, height, width});
    model->to(device);
    JackalModel baseline_model(c10::IntArrayRef{channels, height, width});
    baseline_model->to(device);

    std::unordered_map<std::string, float> config(config_map);
    // TODO tune up hyperparams
    std::unordered_map<std::string, float> default_config{
            {"train_learning_rate",     1e-4},
            {"train_l2_regularization", 0},
            {"train_replay_buffer",     2048},
            {"train_epochs",            1},
            {"train_batch_size",        32},

            {"simulation_cycle_games",  128},
            {"simulation_cycles",       1000},
            {"simulation_temperature",  0.5},
            {"simulation_threads",      64},
            {"simulation_max_turns",    1000},

            {"mcts_iterations",         256},
            {"mcts_exploration",        2},

            {"eval_size",               10},
            {"eval_temperature",        0.1},

            {"timeout",                 600}
    };
    for (auto &kv : default_config) {
        if (config.find(kv.first) == config.end()) {
            config[kv.first] = kv.second;
        }
    }
    Trainer<Jackal, JackalModel> trainer(config, torch::kCUDA);
    auto result = trainer.simulate_and_train(
            model,
            baseline_model,
            nullptr,
            [height, width, players](
                    JackalModel &model,
                    const std::unordered_map<std::string, float> &config
            ) {
                return multithreaded_self_plays(height, width, players, model, config);
            }
    );

    torch::save(model, "models/jackal_model.pt");
    return result;
}