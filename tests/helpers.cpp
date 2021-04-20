#include "helpers.h"
#include "../src/util/utils.h"

#include <mutex>
#include <torch/torch.h>
std::mutex m;



inline void test_srand() {
    srand(123);
    torch::manual_seed(123);
    torch::set_num_threads(1);
//    torch::set_num_interop_threads(1);
    torch::globalContext().setDeterministicAlgorithms(true);
    torch::globalContext().setDeterministicCuDNN(true);
    get_generator() = std::default_random_engine(123);
};

TestGuard::TestGuard() {
    m.lock();
    test_srand();
}

TestGuard::~TestGuard() {
    m.unlock();
}
