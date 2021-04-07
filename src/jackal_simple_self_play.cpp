#include "jackal/jackal.h"
#include "../third_party/queue/concurrentqueue.h"
#include "../third_party/queue/lightweightsemaphore.h"
#include "jackal/game_model.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace torch::indexing;
using namespace std;
using namespace moodycamel;

struct TModelJob {
    torch::Tensor *state;
    torch::Tensor *policy;
    LightweightSemaphore *semaphore;
};

typedef ConcurrentQueue<TModelJob> TModelQueue;

const int WIDTH = 7;
const int HEIGHT = 7;
const int PLAYERS = 2;

static const int NUM_THREADS = 128;

void self_play_thread(TModelQueue *queue) {
    Jackal jackal(HEIGHT, WIDTH, PLAYERS);
    LightweightSemaphore semaphore;
    torch::Tensor policy;// = torch::zeros({7 * 7 * 7 * 7 * 2});
    for (int i = 0; i < 100000; ++i) {
        auto state = jackal.get_state();
        TModelJob item{&state, &policy, &semaphore};
        queue->enqueue(item);
        semaphore.wait();
        jackal = jackal.take_action(policy);
    }
}

void log_model(time_t &tm, int &total_states) {
    time_t tm1;
    time(&tm1);
    if (tm1 - tm > 1) {
        cout << " processed " << total_states << endl;
        total_states = 0;
        tm = tm1;
    }
}

struct RequestContext {
    vector<TModelJob> items;
    torch::Tensor batch;
    GameModelOutput model_output;
};


void reply(RequestContext &request) {
    auto &items(request.items);
    auto &model_output(request.model_output);
    model_output.policy = model_output.policy.to(torch::kCPU);
    for (int i = 0; i < items.size(); ++i) {
        *items[i].policy = model_output.policy.index({i, "..."});
        items[i].semaphore->signal();
    }
}

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
    vector<torch::Tensor> states;
    for (auto &i : items) {
        states.push_back(*i.state);
    }
    request.batch = torch::stack({&states[0], states.size()}).to(torch::kCUDA);
    return terminate;
}


void model_loop(GameModel &model, TModelQueue &queue) {
    time_t tm;
    time(&tm);
    int total_states = 0;
    RequestContext cur_request;
    bool terminate = false;
    while (!terminate) {
        terminate = read_request(queue, cur_request);
        total_states += cur_request.items.size();
        cur_request.model_output = model.forward(cur_request.batch);
        reply(cur_request);
        log_model(tm, total_states);
    }
}


int main() {
    TModelQueue queue;
    std::vector<std::thread> self_play_threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        self_play_threads.emplace_back(std::thread(self_play_thread, &queue));
    }
    Jackal jackal(HEIGHT, WIDTH, PLAYERS);
    auto state = jackal.get_state();
    torch::NoGradGuard no_grad;
    JackalModel model(state.sizes(), 128, 10, 2);
    model->to(torch::kCUDA);
    model->eval();
    model_loop(*model, queue);
}