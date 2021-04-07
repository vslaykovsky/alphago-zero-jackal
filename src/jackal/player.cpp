#include "player.h"

using namespace torch::indexing;

Player::Player(int player_idx) : player_idx(player_idx) {

}

Player::Player(int player_idx, int w, int h, bool render, bool debug) :
        GameElement(torch::zeros({PLAYER_PLANES_NUMBER, h, w}), render, debug),
        player_idx(player_idx) {
    Coords coords = std::vector<Coords>{
            {0,           height() / 2},
            {width() - 1, height() / 2},
            {width() / 2, 0},
            {width() / 2, height() - 1}
    }[player_idx];
    state[PLANE_SHIP][coords.y][coords.x] = 1;
    state[PLANE_PIRATES][coords.y][coords.x] = 3;
}

void Player::load(const torch::Tensor &tensor) {
    state = tensor;
}

void Player::set_current_player(bool current_player) {
    state.index_put_({PLANE_CURRENT_PLAYER, "..."}, (int) current_player);
}

bool Player::is_current_player() const {
    return state[PLANE_CURRENT_PLAYER][0][0].item<bool>();
}

void Player::inc_score(int score) {
    state.index_put_({PLANE_SCORE, "..."}, get_score() + score);
}

cv::Mat Player::get_image() {
    static std::vector<cv::Scalar> player_color = {
            cv::Scalar(255, 255, 255, 255),
            cv::Scalar(0, 0, 0, 255)
    };
    cv::Mat image(height() * TILE_SIZE, width() * TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    auto ship = get_ship_coords();
    render_tile(image, ship.x, ship.y, SpriteType::SHIP1);

    auto pirates = get_pirate_coords();
    for (auto &p : pirates) {
        auto center = cv::Point(int((p.x + 0.75) * TILE_SIZE), int((p.y + 0.25) * TILE_SIZE));
        cv::circle(image,
                   center,
                   TILE_SIZE / 4,
                   player_color[player_idx], cv::FILLED);
        int n = state[PLANE_PIRATES][p.y][p.x].item<int>();
        cv::putText(image, std::to_string(n), center, cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(128, 128, 128, 255), 2);
    }
    cv::putText(image, "Score:" + std::to_string(get_score()),
                cv::Point(ship.x * TILE_SIZE, int((ship.y + 0.9) * TILE_SIZE)),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255, 255), 2);
    return image;
}

int Player::get_score() const {
    return state[PLANE_SCORE][0][0].item<int>();
}

Coords Player::get_ship_coords() const {
    int idx = state[PLANE_SHIP].argmax().item<int>();
    int y = idx / width();
    int x = idx % width();
    return {x, y};
}

void Player::move_ship(const Coords &to) {
    auto from = get_ship_coords();
    state[PLANE_SHIP][from.y][from.x] -= 1;
    state[PLANE_SHIP][to.y][to.x] += 1;
}

void Player::remove_pirate(const Coords &from, bool all) {
    if (all) {
        state[PLANE_PIRATES][from.y][from.x] = 0;
    } else {
        state[PLANE_PIRATES][from.y][from.x] -= 1;
    }
}

void Player::move_pirate(const Coords &from, const Coords &to, bool all) {
    int n = 1;
    if (all)
        n = state[PLANE_PIRATES][from.y][from.x].item<int>();
    remove_pirate(from, all);
    state[PLANE_PIRATES][to.y][to.x] += n;
}

std::vector<Coords> Player::get_pirate_coords() const {
    std::vector<Coords> coords;
    auto pirates = state.index({PLANE_PIRATES, "..."}).nonzero();
    for (int i = 0; i < pirates.size(0); ++i) {
        Coords p(pirates[i][1].item<int>(), pirates[i][0].item<int>());
        coords.push_back(p);
    }
    return coords;
}

void collect_destinations(
        const Coords &from, const Coords &to, int turns, const Coords &ship,
        std::unordered_set<Coords> &visited, std::unordered_set<Action> &actions,
        const Ground &ground
) {
    if (turns < 0 || visited.find(to) != visited.end()) {
        return;
    }
    visited.insert(to);
    bool is_arrow = ground.is_arrow(to.x, to.y);
    if (from != to && !is_arrow) {
        if (ground.get_gold(from.x, from.y) > 0) {
            actions.insert(Action(from, to, true));
        }
        actions.insert(Action(from, to, false));
    }
    std::vector<Coords> directions = ground.get_directions(to.x, to.y);
    if (!is_arrow) {
        if (ship == to) {
            // from ship
            for (auto &d : Ground::all_directions(false)) {
                if (ground.is_ground(to.x + d.x, to.y + d.y)) {
                    directions.push_back(d);
                    break;
                }
            }
        } else if (abs(ship.x - to.x) <= 1 && abs(ship.y - to.y) <= 1) {
            // to ship
            directions.emplace_back(ship.x - to.x, ship.y - to.y);
        }
    }
    for (auto &dir : directions) {
        collect_destinations(
                from,
                Coords(to.x + dir.x, to.y + dir.y),
                turns - ground.get_delay(to.x, to.y),
                ship,
                visited,
                actions,
                ground);
    }
}

std::unordered_set<Action> Player::get_pirate_actions(const Coords &c, const Ground &ground) const {
    std::unordered_set<Action> actions;
    std::unordered_set<Coords> visited;
    collect_destinations(c, c, 1, get_ship_coords(), visited, actions, ground);
    return actions;
}

std::unordered_set<Action> Player::get_ship_actions() const {
    std::unordered_set<Action> actions;
    auto from = get_ship_coords();
    if (from.x == 0 || from.x == width() - 1) {
        if (from.y > 1)
            actions.emplace(Action(from, Coords(from.x, from.y - 1), false));
        if (from.y < height() - 2)
            actions.emplace(Action(from, Coords(from.x, from.y + 1), false));
    } else if (from.y == 0 || from.y == height() - 1) {
        if (from.x > 1)
            actions.emplace(Action(from, Coords(from.x - 1, from.y), false));
        if (from.x < width() - 2)
            actions.emplace(Action(from, Coords(from.x + 1, from.y), false));
    }
    return actions;
}

std::vector<Action> Player::get_possible_actions(const Ground &ground) const {
    const std::vector<Coords> &pirate_coords = get_pirate_coords();
    if (pirate_coords.empty()) {
        return std::vector<Action>();
    }
    auto actions = get_ship_actions();
    for (auto &pirate: pirate_coords) {
        auto pirate_actions = get_pirate_actions(pirate, ground);
        actions.insert(pirate_actions.begin(), pirate_actions.end());
    }
    return std::vector<Action>(actions.begin(), actions.end());
}

int Player::get_pirates(const Coords& p) const {
    return state[PLANE_PIRATES][p.y][p.x].item<int>();
}
