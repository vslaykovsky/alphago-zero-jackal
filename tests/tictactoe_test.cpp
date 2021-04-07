#include <gtest/gtest.h>
#include "../src/tictactoe/tictactoe_model.h"
#include "../src/rl/self_play.h"
#include "../src/rl/train.h"
#include "../src/tictactoe/tictactoe.h"

using namespace std;



TEST(TTTSelfPlay, TestTrainedModel) {
    TicTacToeModel model;
    TicTacToeModel rand_model;
    torch::load(model, "models/tictactoe_model.pt");
    StateValueDataset<TicTacToe> ds(10);
    for (int i = 0; i < *ds.size(); ++i) {
        auto ex = ds.get(i);
        auto out = model(ex.data);
        cout << ex.data << endl << out.value << endl << ex.target << endl;
    }
    cout << "trained model 'x'" << endl;
    mcts_model_self_play<TicTacToe>(model, rand_model, 256, 0.1, 2, torch::kCPU, true);
    cout << "trained model 'o'" << endl;
    mcts_model_self_play<TicTacToe>(rand_model, model, 256, 0.1, 2, torch::kCPU, true);

    cout << "compare to random " << compare_models<TicTacToe>(model, rand_model, 256, 0.1, 1) << endl;
}


TEST(TTTSelfPlay, TestStateValueDataSet) {
    std::srand(0);
    StateValueDataset<TicTacToe> ds(1);
    ASSERT_EQ(" 1\n-1\n 1\n-1\n 1\n 1\n 1\n 1\n-1\n-1\n[ CPUFloatType{10} ]", to_string(ds.states[0]));
    ASSERT_EQ(" 1\n-1\n[ CPUFloatType{2} ]", to_string(ds.state_values[0]));
}

TEST(TTTSelfPlay, TestSelfPlayDataSet) {
    TicTacToeModel model;
    srand(0);
    auto self_play = mcts_model_self_play<TicTacToe, TicTacToeModel>(model, 1, 1., 1.);
    auto ds = SelfPlayDataset(std::vector<SelfPlayResult>{self_play}, 1, false);
    auto ex = ds.examples.back();
    ASSERT_EQ(" 1 -1  1 -1  1  1  1  1 -1 -1\n[ CPUFloatType{1,10} ]", to_string(ex.x));
    ASSERT_EQ(" 1 -1\n[ CPUFloatType{1,2} ]", to_string(ex.state_value));
    ASSERT_EQ(" 5\n[ CPULongType{1} ]", to_string(ex.action_proba));
}


TEST(TTTSelfPlay, TestFitTerminalStates) {
    TicTacToeModel random_model;
    TicTacToeModel model;
    StateValueDataset<TicTacToe> eval_ds(100);

    vector<SelfPlayResult> self_plays;
    for (int i = 0; i < 1024; ++i) {
        self_plays.push_back(mcts_model_self_play<TicTacToe, TicTacToeModel>(model, 1, 1., 1.));
    }
    SelfPlayDataset ds(self_plays, 32, true, true);

    int step = 0;
    Trainer<TicTacToe, TicTacToeModel> trainer;
    trainer.train(model, ds, random_model, &eval_ds, step);
    float trained_loss = evaluate<TicTacToe, TicTacToeModel>(model, eval_ds);
    float random_loss = evaluate<TicTacToe, TicTacToeModel>(random_model, eval_ds);
    cerr << "trained_loss " << trained_loss << " random_loss " << random_loss << endl;
    ASSERT_LT(trained_loss, random_loss);
    ASSERT_LT(trained_loss, 0.5); // must fit well
    ASSERT_GT(random_loss, 1);
}