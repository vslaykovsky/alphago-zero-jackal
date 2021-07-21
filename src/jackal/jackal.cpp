#include <random>
#include "jackal.h"
#include "game_model.h"

Jackal::Jackal(int height, int width, int players_num, bool render, bool debug) :
        ground(height, width, render, debug),
        current_player(rand() % players_num),
        turn(0),
        render(render),
        debug(debug) {
    std::cout << "start player " << current_player << std::endl;
    for (int p = 0; p < players_num; ++p) {
        players.emplace_back(Player(p, width, height, render, debug));
        if (p == current_player) {
            players[p].set_current_player(true);
        }
    }
}


Jackal Jackal::take_action(int action) const {
    return take_action(decode_action(action));
}

Jackal Jackal::take_action(const Action &action) const {
    Jackal j(*this);
    auto &player = j.players[j.current_player];
    const std::unordered_set<Action> &ship_action = player.get_ship_actions();
    if (ship_action.find(action) != ship_action.end()) {
        // ship action
        player.move_ship(action.coordinates_to);
        player.move_pirate(action.coordinates_from, action.coordinates_to, true);
        for (auto &p : j.players) {
            // kill all enemies
            if (&p != &player) {
                p.remove_pirate(action.coordinates_to, true);
            }
        }
    } else {
        // pirate action

        // killed by an enemy ship
        for (auto &p : j.players) {
            if (&p != &player) {
                if (action.coordinates_to == p.get_ship_coords()) {
                    player.remove_pirate(action.coordinates_from);
                    goto end_of_action;
                }
            }
        }

        // move
        player.move_pirate(action.coordinates_from, action.coordinates_to);
        if (action.with_items) {
            // move gold
            if (action.coordinates_to == player.get_ship_coords()) {
                // gold to the ship
                player.inc_score();
                j.ground.remove_gold(action.coordinates_from);
            } else if (ground.is_ground(action.coordinates_to.x, action.coordinates_to.y)) {
                // ground
                j.ground.move_gold(action.coordinates_from, action.coordinates_to);
            } else {
                // water
                j.ground.remove_gold(action.coordinates_from);
            }
        }

        // all enemies to their ship
        for (auto &p : j.players) {
            // kill all enemies
            if (&p != &player && p.get_pirates(action.coordinates_to)) {
                p.move_pirate(action.coordinates_to, p.get_ship_coords(), true);
            }
        }
    }
    end_of_action:
    j.set_next_player();
    return j;
}

std::vector<int> Jackal::get_possible_actions() const {
    const std::vector<Action> &actions = players[current_player].get_possible_actions(ground);
    std::vector<int> action_codes;
    for (auto &a: actions) {
        action_codes.push_back(encode_action(a));
    }
    return action_codes;
}

void Jackal::store(const std::string &file_name) const {
    torch::Tensor state(get_state());
    torch::save(state, file_name);
}


cv::Mat Jackal::get_image(MCTSStateActionValue *mcts) {
    if (!render) {
        throw std::runtime_error("get_image() is called from an object that doesn't support rendering");
    }
    auto ground_img = ground.get_image().clone();
    for (int i = 0; i < players.size(); ++i) {
        auto &player = players[i];
        auto player_img = player.get_image(mcts ? mcts->state_value[i] : 0.);
        copy_with_alpha(ground_img, player_img, 0, 0);
    }
    if (debug) {
        auto actions = get_possible_actions();
        for (auto code : actions) {
            auto action = decode_action(code);
            cv::arrowedLine(ground_img, tile_center(action.coordinates_from), tile_center(action.coordinates_to),
                            cv::Scalar(255, 255, 255), 5);
            if (mcts) {
                float proba = mcts->action_proba[code];
                cv::putText(ground_img, std::to_string(proba).substr(0, 4), tile_center(action.coordinates_to),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 255), 2);
            }
        }
    }
    cv::putText(ground_img, "Turn: " + std::to_string(turn), cv::Point(ground_img.cols / 2 - TILE_SIZE, TILE_SIZE / 2),
                cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255, 255), 3);
    return ground_img;
}

void Jackal::load(torch::Tensor &state) {
    int plane = 0;
    ground.load(state.index({Slice(0, GROUND_PLANES_NUMBER), "..."}));
    plane += GROUND_PLANES_NUMBER;
    int player_idx = 0;
    turn = 0;
    while (plane < state.size(0)) {
        Player p(player_idx);
        p.load(state.index({Slice(plane, plane + PLAYER_PLANES_NUMBER), "..."}));
        plane += PLAYER_PLANES_NUMBER;
        if (p.is_current_player()) {
            current_player = player_idx;
        }
        ++player_idx;
    }
}

void Jackal::load(const std::string &file_name) {
    torch::Tensor state;
    torch::load(state, file_name);
    load(state);
}

torch::Tensor Jackal::get_state() const {
    std::vector<torch::Tensor> l;
    l.push_back(ground.state);
    for (auto &player: players) {
        l.push_back(player.state);
    }
    return torch::unsqueeze(torch::cat({&l[0], l.size()}, 0), 0);
}

void Jackal::set_next_player() {
    players[current_player].set_current_player(false);
    current_player = int((current_player + 1) % players.size());
    players[current_player].set_current_player(true);
    ++turn;
}

int Jackal::get_random_action() const {
    auto actions = get_possible_actions();
    if (actions.empty()) {
        throw std::runtime_error("Empty action set");
    }
    return actions[rand() % actions.size()];
}


Jackal Jackal::take_action(const torch::Tensor &taction, float temperature) {
    auto possible_actions_idx = encode_possible_actions();
    auto action_proba = taction.index({possible_actions_idx});
    action_proba = (action_proba / std::max(action_proba.sum().item().toDouble(), 1e-8)).pow(1 / temperature);


    std::discrete_distribution<int> distribution(action_proba.data_ptr<float>(),
                                                 action_proba.data_ptr<float>() + action_proba.size(0));
    int action_idx = distribution(generator);
    return take_action(get_possible_actions()[action_idx]);
}

int Jackal::encode_action(const Action &action) const {
    std::vector<std::pair<int, int>> elements{
            {action.coordinates_from.y, height()},
            {action.coordinates_from.x, width()},
            {action.coordinates_to.y,   height()},
            {action.coordinates_to.x,   width()},
            {action.with_items,         2},
    };
    int code = 0;
    for (auto element : elements) {
        code *= element.second;
        code += element.first;
    }
    return code;
}

Action Jackal::decode_action(int code) const {
    Action action;
    std::vector<std::pair<int *, int>> fields{
            {&action.with_items,         2},
            {&action.coordinates_to.x,   width()},
            {&action.coordinates_to.y,   height()},
            {&action.coordinates_from.x, width()},
            {&action.coordinates_from.y, height()},
    };
    for (auto field : fields) {
        *field.first = code % field.second;
        code /= field.second;
    }
    return action;
}


torch::Tensor Jackal::encode_possible_actions() const {
    const auto &actions = get_possible_actions();
    return torch::tensor(c10::ArrayRef<int>(&actions[0], actions.size()));
}

bool Jackal::is_terminal() {
    return players[current_player].get_pirate_coords().empty() || ground.total_gold() == 0;
}

MCTSStateValue Jackal::get_reward() {
    MCTSStateValue reward;
    int max_score = -1;
    int max_player = -1;
    for (int i = 0; i < players.size(); ++i) {
        int score = players[i].get_score();
        if (score > max_score) {
            max_score = score;
            max_player = i;
        }
    }
    reward.resize(players.size());
    if (max_score > 0) {
        // we have a winner
        for (int i = 0; i < players.size(); ++i) {
            reward[i] = max_player == i ? 1 : -1;
        }
    }
    return reward;
}

int Jackal::get_current_player_id() const {
    return current_player;
}


std::ostream &operator<<(std::ostream &os, const Jackal &j) {
    os << "Jackal(" << j.width() << "," << j.height() << "," << j.players.size() << ")";
    return os;
}