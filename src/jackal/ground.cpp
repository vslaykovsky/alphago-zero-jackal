#include "ground.h"

using namespace torch::indexing;

Ground::Ground(bool render, bool debug) : GameElement(render, debug) {
}



Ground::Ground(int height, int width, bool render, bool debug) :
        GameElement(std::move(torch::zeros({GROUND_PLANES_NUMBER, height, width}, torch::kInt8)), render, debug),
        image(height * TILE_SIZE, width * TILE_SIZE, CV_8UC3) {
    if (render) {
        // "And the Spirit of God moved upon the face of the waters"
        cv::rectangle(image, cv::Point(0, 0), cv::Point(TILE_SIZE * width, TILE_SIZE * height),
                      cv::Scalar(255, 0, 0, 255), cv::FILLED);
    }
    state.index_put_({PLANE_GROUND,
                      Slice(1, height - 1),
                      Slice(1, width - 1)
                     }, 1);
    state.index_put_({PLANE_DELAYS, None, None}, 1);
    static const std::vector<SpriteType> arrows = {SpriteType::ARROW_R,
                                                   SpriteType::ARROW_RU,
                                                   SpriteType::ARROW_R_L,
                                                   SpriteType::ARROW_LB_RU,
                                                   SpriteType::ARROW_LU_R_B,
                                                   SpriteType::ARROW_L_R_U_B,
                                                   SpriteType::ARROW_LU_RU_LB_RB};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            set_default_directions(x, y);
            render_arrows(y, x);
            if (state[PLANE_GROUND][y][x].item<int>() == 1) {
                render_tile(image, x, y, SpriteType::EMPTY1);
                render_arrows(y, x);
                if (rand01() < 0.2) {
                    set_arrow(x, y, arrows[rand() % arrows.size()], rand() % 4);
                } else if (rand01() < 0.2) {
                    set_gold(x, y, rand() % 5 + 1);
                }
            }
        }
    }
}

void Ground::load(const torch::Tensor &tensor) {
    state = tensor;
}

void Ground::render_arrows(int y, int x) {
    if (debug) {
        for (int dir = 0; dir < 8; ++dir) {
            if (state[PLANE_DIRECTIONS + dir][y][x].item<int>() == 1) {
                auto direction = all_directions()[dir];
                cv::arrowedLine(image, {int((x + 0.5) * TILE_SIZE), int((y + 0.5) * TILE_SIZE)},
                                {int((x + 0.5 + direction.x / 3.0) * TILE_SIZE),
                                 int((y + 0.5 + direction.y / 3.0) * TILE_SIZE)},
                                cv::Scalar(0, 255, 255), 3);
            }
        }
    }
}


int Ground::get_gold(int x, int y) const {
    return state[PLANE_GOLD][y][x].item<int>();
}

void Ground::set_gold(int x, int y, int coins) {
    state[PLANE_GOLD][y][x] = coins;
    static std::vector<SpriteType> golds = {GOLD1, GOLD2, GOLD3, GOLD4, GOLD5};
    render_tile(image, x, y, golds[coins - 1]);
    render_arrows(y, x);
}

void Ground::set_default_directions(int x, int y) {
    auto v = state.index({PLANE_GROUND, y, x}).item<int>();
    auto &all_dirs = all_directions();
    auto directions = torch::zeros({(long) all_dirs.size()});
    for (int i = 0; i < all_dirs.size(); ++i) {
        auto d = all_dirs[i];
        int y1 = y + d.y;
        int x1 = x + d.x;
        if (x1 >= 0 && x1 < width() && y1 >= 0 && y1 < height()) {
            int v1 = state[PLANE_GROUND][y1][x1].item<int>();
            if (v1 == v) {
                directions[i] = 1;
            }
        }
    }
    state.index_put_({Slice(PLANE_DIRECTIONS, PLANE_DIRECTIONS + 8), y, x}, directions);
}

void Ground::set_arrow(int x, int y, SpriteType arrow_type, int rotation) {
    auto arrow = encode_arrow(arrow_type, rotation);
    state.index_put_({Slice(PLANE_DIRECTIONS, PLANE_DIRECTIONS + 8), y, x}, arrow);
    state[PLANE_DELAYS][y][x] = 0;
    render_tile(image, x, y, arrow_type, rotation);
    render_arrows(y, x);
}

bool Ground::is_ground(int x, int y) const {
    return valid_coord(x, y) && state[PLANE_GROUND][y][x].item<int>() != 0;
}

bool Ground::is_arrow(int x, int y) const {
    return state[PLANE_DELAYS][y][x].item<int>() == 0;
}

int Ground::get_delay(int x, int y) const {
    return state[PLANE_DELAYS][y][x].item<int>();
}

void Ground::set_delay(int x, int y, int delay) {
    state[PLANE_DELAYS][y][x] = delay;
}

std::vector<Coords> Ground::get_directions(int x, int y) const {
    std::vector<Coords> result;
    auto &dirs = all_directions();
    for (int d = 0; d < dirs.size(); ++d) {
        if (state[PLANE_DIRECTIONS + d][y][x].item<int>() == 1) {
            result.push_back(dirs[d]);
            assert(valid_coord(x + dirs[d].x, y + dirs[d].y));
        }
    }
    return result;
}

const std::vector<Coords> &Ground::get_arrow_directions(SpriteType st) {
    static std::unordered_map<SpriteType, std::vector<Coords>> arrow_directions = {
            {SpriteType::ARROW_R,           {{1,  0}}},
            {SpriteType::ARROW_RU,          {{1,  -1}}},
            {SpriteType::ARROW_R_L,         {{1,  0},  {-1, 0}}},
            {SpriteType::ARROW_LB_RU,       {{-1, 1},  {1,  -1}}},
            {SpriteType::ARROW_LU_R_B,      {{-1, -1}, {1,  0},  {0, 1}}},
            {SpriteType::ARROW_L_R_U_B,     {{0,  1},  {-1, 0},  {0, -1}, {1, 0}}},
            {SpriteType::ARROW_LU_RU_LB_RB, {{-1, 1},  {-1, -1}, {1, -1}, {1, 1}}},
    };
    return arrow_directions[st];
}

torch::Tensor Ground::encode_arrow(SpriteType arrow, int rotation) {
    auto &all_dir = all_directions();
    auto result = torch::zeros(all_dir.size());
    for (auto &coord: get_arrow_directions(arrow)) {
        auto c(coord);
        for (int r = 0; r < rotation; ++r) {
            c = Coords(-c.y, c.x);
        }
        int idx = std::find(all_dir.begin(), all_dir.end(), c) - all_dir.begin();
        result[idx] = 1;
    }
    return result;
}

cv::Mat Ground::get_image() {
    auto img = image.clone();
    auto gold_coords = state.index({PLANE_GOLD, "..."}).nonzero();
    for (int i = 0; i < gold_coords.size(0); ++i) {
        Coords p(gold_coords[i][1].item<int>(), gold_coords[i][0].item<int>());
        Coords center(int((p.x + 0.25) * TILE_SIZE), int((p.y + 0.25) * TILE_SIZE));
        cv::circle(img, center, TILE_SIZE / 4,
                   cv::Scalar(0, 255, 255), cv::FILLED);
        int num = state[PLANE_GOLD][p.y][p.x].item<int>();
        cv::putText(img, std::to_string(num), center, cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(128, 128, 128, 255), 2);
    }
    return img;
}


void Ground::move_gold(const Coords &from, const Coords &to) {
    state[PLANE_GOLD][from.y][from.x].sub_(1);
    state[PLANE_GOLD][to.y][to.x].add_(1);
}

int Ground::total_gold() {
    return state[PLANE_GOLD].sum().item().toInt();
}

void Ground::remove_gold(Coords point) {
    state[PLANE_GOLD][point.y][point.x].sub_(1);
}

bool Ground::valid_coord(int x, int y) const {
    return x >= 0 && x < width() && y >= 0 && y < height();
}

void Ground::set_ground(int x, int y) {
    set_default_directions(x, y);
    set_delay(x, y, 1);
}
