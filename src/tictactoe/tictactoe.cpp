#include <numeric>
#include "tictactoe.h"


using namespace std;

TicTacToe::TicTacToe(const vector<vector<int>> &field) : turn(0), field(field) {

}

TicTacToe::TicTacToe(int size) : turn(0) {
    field.resize(size);
    for (auto &f : field) {
        f.resize(size);
    }
}

int TicTacToe::win() const {
    int size = field.size();
    for (auto &row : field) {
        int acc = accumulate(row.begin(), row.end(), 0);
        if (acc == size) {
            return 1;
        }
        if (acc == -size)
            return -1;
    }
    for (int x = 0; x < size; x++) {
        int acc = 0;
        for (int y = 0; y < size; y++) {
            acc += field[y][x];
        }
        if (acc == size) {
            return 1;
        }
        if (acc == -size)
            return -1;
    }
    int acc = 0;
    for (int i = 0; i < size; i++) {
        acc += field[i][i];
    }
    if (acc == size) {
        return 1;
    }
    if (acc == -size)
        return -1;
    acc = 0;
    for (int i = 0; i < size; i++) {
        acc += field[size - 1 - i][i];
    }
    if (acc == size) {
        return 1;
    }
    if (acc == -size)
        return -1;
    return 0;
}

MCTSStateValue TicTacToe::get_reward() const {
    MCTSStateValue value;
    value.resize(2);
    int w = win();
    value[0] = 0.;
    value[1] = 0.;
    if (w == 1) {
        value[0] = 1;
        value[1] = -1;
    }
    if (w == -1) {
        value[0] = -1;
        value[1] = 1;
    }
    return value;
}

vector<int> TicTacToe::get_possible_actions() const {
    vector<int> result;
    if (win() != 0) {
        return result;
    }
    int i = 0;
    for (auto &row: field) {
        for (auto v : row) {
            if (v == 0) {
                result.push_back(i);
            }
            ++i;
        }
    }
    return result;
}

TicTacToe TicTacToe::take_action(int action) const {
    int size = field.size();
    TicTacToe state(*this);
    int &v = state.field[action / size][action % size];
    if (v != 0) {
        throw std::runtime_error("Invalid action");
    }
    v = turn % 2 == 0 ? 1 : -1;
    state.turn++;
    return state;
}

int TicTacToe::get_current_player_id() const {
    return turn % 2;
}

torch::Tensor TicTacToe::get_state() const {
    int size = field.size();
    vector<float> v;
    v.push_back(get_current_player_id());
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            v.push_back(field[y][x]);
        }
    }
    return torch::from_blob(&v[0], {(int) v.size()}, torch::kFloat).unsqueeze(0).clone();
}

cv::Mat TicTacToe::get_image() const {
    throw std::runtime_error("not supported");
}


std::ostream &operator<<(std::ostream &os, const TicTacToe &ttt) {
    for (auto &row : ttt.field) {
        os << "|";
        for (auto v : row) {
            char codes[] = "o x";
            os << codes[v + 1] << "|";
        }
        os << endl;
    }
    return os;
}