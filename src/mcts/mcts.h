#pragma once

#include <unordered_map>
#include <cmath>
#include <iostream>
#include <functional>
#include <vector>
#include <random>
#include <memory>
#include <tensorboard_logger.h>

#include "../util/utils.h"


typedef std::vector<float> MCTSStateValue;
typedef std::unordered_map<int, float> MCTSActionValue;

struct MCTSStateActionValue {
    MCTSStateValue state_value;
    MCTSActionValue action_proba;

    int sample_action(float temperature = 1.0) const {
        std::vector<float> values;
        std::vector<int> actions;
        for (auto &kv: action_proba) {
            values.push_back(pow(kv.second, 1 / temperature));
            actions.push_back(kv.first);
        }
        std::discrete_distribution<int> distribution(&values[0], &values[0] + values.size());
        int action_idx = distribution(get_generator());
        return actions[action_idx];
    }

    int best_action() const {
        std::vector<int> actions;
        float max_proba = 0;
        for (auto &kv: action_proba) {
            if (kv.second > max_proba) {
                max_proba = kv.second;
                actions = std::vector<int>({kv.first});
            } else if (kv.second == max_proba) {
                actions.push_back(kv.first);
            }
        }
        return actions[rand() % actions.size()];
    }

    void log(TensorBoardLogger *logger, int step, float temperature) {
        std::vector<float> values;
        float max_proba = 0;
        float proba_sum = 0;
        for (auto& kv: action_proba) {
            float p = pow(kv.second, temperature);
            values.push_back(p);
            if (p > max_proba) {
                max_proba = p;
            }
            proba_sum += p;
        }
        std::sort(values.begin(), values.end());
        logger->add_histogram("mcts_action_dist", step, values);
        logger->add_scalar("mcts_best_action_proba", step, max_proba / proba_sum);
        logger->add_scalar("mcts_first_player_value", step, state_value[0]);
        logger->add_scalar("mcts_second_player_value", step, state_value[1]);
    }
};


template<class T, class F>
struct MCTSNode {
    T state;
    MCTSNode<T, F> *parent;
    std::unordered_map<int, std::unique_ptr<MCTSNode<T, F>>> children;

    MCTSStateActionValue prior;
    MCTSStateValue state_value_sum;
    int visits;

    std::vector<int> possible_actions;
    bool is_terminal;

    MCTSNode(const T &pstate, MCTSNode<T, F> *parent, F value_func)
            : state(pstate),
              visits(0),
              possible_actions(pstate.get_possible_actions()),
              is_terminal(possible_actions.empty()),
              prior(value_func(pstate)),
              parent(parent) {
    }

    MCTSStateValue mean_state_values() const {
        MCTSStateValue result(state_value_sum);
        for (auto &v : result) {
            v /= (float) visits;
        }
        return result;
    }

    MCTSActionValue action_proba() {
        MCTSActionValue value;
        float sum = 0.;
        for (auto &kv: children) {
            int v = kv.second.get() ? kv.second->visits : 0;
            value[kv.first] = v;
            sum += (float) v;
        }
        for (auto &kv: value) {
            kv.second /= sum;
        }
        return value;
    }
};

template<class T, class F>
std::pair<float, float> mcts_action_value(MCTSNode<T, F> &node, int action, float exploration) {
    auto child = node.children[action].get();
    auto child_state_value = child ? child->state_value_sum[node.state.get_current_player_id()] /
                                     (float) child->visits
                                   : 0.;
    auto prior_action_proba = node.prior.action_proba[action];
    auto uct = sqrt(node.visits) / (1 + (child ? child->visits : 0));
    return std::pair<float, float>(child_state_value, prior_action_proba * uct * exploration); // alphago zero paper - PUCT
}


template<class T, class F>
int mcts_best_action(MCTSNode<T, F> &node, float exploration) {
    if (node.is_terminal) {
        throw std::runtime_error("mcts_best_action called for a terminal state");
    }
    float best_value = -1000;
    std::vector<int> best_actions;
    for (int action : node.possible_actions) {
        auto action_value = mcts_action_value(node, action, exploration);
        auto value = action_value.first + action_value.second;
        if (value > best_value) {
            best_value = value;
            best_actions.clear();
            best_actions.push_back(action);
        } else if (value == best_value) {
            best_actions.push_back(action);
        }
    }
    if (best_actions.empty()) {
        throw std::runtime_error("couldn't find best action");
    }
    int best_action = best_actions[rand() % best_actions.size()];
    return best_action;
}

template<class T, class F>
MCTSNode<T, F> &
mcts_select(MCTSNode<T, F> &node, F value_func, float exploration) {
    if (node.is_terminal) {
        return node;
    }
    int best_action = mcts_best_action(node, exploration);
    if (node.children[best_action].get()) {
        return mcts_select(*node.children[best_action], value_func, exploration);
    } else {
        node.children[best_action].reset(new MCTSNode<T, F>(node.state.take_action(best_action), &node, value_func));
        return *node.children[best_action];
    }
}

template<class T, class F>
void back_propagate(MCTSNode<T, F> &leaf) {
    auto node = &leaf;
    while (node) {
        node->state_value_sum.resize(leaf.prior.state_value.size());
        for (int i = 0; i < leaf.prior.state_value.size(); ++i) {
            node->state_value_sum[i] += leaf.prior.state_value[i];
        }
        node->visits++;
        node = node->parent;
    }
}


template<class T, class F>
MCTSStateActionValue mcts_search(
        const T &state, F value_func, int iterations, float exploration, TensorBoardLogger* logger = nullptr, int step=0) {
    MCTSNode<T, F> root(state, nullptr, value_func);
    for (int i = 0; i < iterations; ++i) {
        auto &node = mcts_select(root, value_func, exploration);
        back_propagate(node);
    }
    if (logger) {
        std::vector<float> mcts_mean_state_value;
        std::vector<float> mcts_exploration;
        for (int action : root.possible_actions) {
            auto action_value = mcts_action_value(root, action, exploration);
            mcts_mean_state_value.push_back(action_value.first);
            mcts_exploration.push_back(action_value.second);
        }
        logger->add_histogram("mcts_mean_state_value", step, mcts_mean_state_value);
        logger->add_histogram("mcts_exploration", step, mcts_exploration);
    }
    return MCTSStateActionValue{root.mean_state_values(), root.action_proba()};
}
