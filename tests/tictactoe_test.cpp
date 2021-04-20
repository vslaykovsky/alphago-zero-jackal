#include <gtest/gtest.h>
#include "../src/tictactoe/tictactoe_model.h"
#include "../src/rl/self_play.h"
#include "../src/rl/train.h"
#include "../src/tictactoe/tictactoe.h"
#include "helpers.h"
using namespace std;


TEST(MCTS, TicTacToe) {
    TestGuard g;
    TicTacToe ttt;
    while (!ttt.get_possible_actions().empty()) {
        const vector<int> &actions = ttt.get_possible_actions();
        ttt = ttt.take_action(actions[rand() % actions.size()]);
    }
    ASSERT_EQ(ttt.get_reward()[0], 0);
}

TEST(TTTSelfPlay, TestStateValueDataSet) {
    TestGuard g;


    TicTacToeModel model;
    TicTacToe game;
    vector<SelfPlayResult> self_plays = {mcts_model_self_play<TicTacToe, TicTacToeModel>(
            game, model, model, 256, 10, 1., 1.
    )};
    SelfPlayDataset ds(self_plays, 1, false);
    auto ex = ds.get(ds.size() - 1);
    ASSERT_EQ(" 1  1  1 -1 -1  1 -1 -1  1  1\n[ CPUFloatType{1,10} ]", to_string(ex.x));
    // " 0  1  1 -1  1  1 -1  0 -1 -1\n[ CPUFloatType{1,10} ]"
    ASSERT_EQ(" 1 -1\n[ CPUFloatType{1,2} ]", to_string(ex.state_value));
}

TEST(TTTSelfPlay, TestSelfPlayDataSet) {
    TestGuard g;

    TicTacToeModel model;

    TicTacToe game;
    auto self_play = mcts_model_self_play<TicTacToe, TicTacToeModel>(game, model, model, 1, 10, 1., 1.);
    auto ds = SelfPlayDataset(std::vector<SelfPlayResult>{self_play}, 1, false);
    auto ex = ds.examples.back();
    ASSERT_EQ(" 1 -1 -1  1  1  1 -1 -1  1  1\n[ CPUFloatType{1,10} ]", to_string(ex.x));
    ASSERT_EQ(" 0  0\n[ CPUFloatType{1,2} ]", to_string(ex.state_value));
    ASSERT_EQ(" 3\n[ CPULongType{1} ]", to_string(ex.action_proba));
}


TEST(TTTSelfPlay, FullTrainingCycle) {
    TestGuard g;

    // train

    {
        TicTacToeModel random_model;
        TicTacToeModel model;

        vector<SelfPlayResult> self_plays;
        for (int i = 0; i < 500; ++i) {
            TicTacToe game;
            self_plays.push_back(mcts_model_self_play<TicTacToe, TicTacToeModel>(
                    game, model, model, 1, 10, 1., 1.
            ));
        }
        SelfPlayDataset train_ds(self_plays, 1, true, torch::kCPU, false);

        self_plays.clear();
        for (int i = 0; i < 100; ++i) {
            TicTacToe game;
            self_plays.push_back(mcts_model_self_play<TicTacToe, TicTacToeModel>(
                    game, model, model, 1, 10, 1., 1.
            ));
        }
        SelfPlayDataset eval_ds(self_plays, 1, true, torch::kCPU, true);


        int step = 0;
        Trainer<TicTacToe, TicTacToeModel> trainer({
                                                           {"train_learning_rate", 1e-3},
                                                           {"train_replay_buffer", 10024},
                                                           {"train_epochs",        10},
                                                           {"train_batch_size",    32}
                                                   });
        trainer.train(model, train_ds, random_model, &eval_ds, step);
        float trained_loss = evaluate<TicTacToeModel>(model, eval_ds);
        float random_loss = evaluate<TicTacToeModel>(random_model, eval_ds);
        cerr << "trained_loss " << trained_loss << " random_loss " << random_loss << endl;
        ASSERT_LT(trained_loss, random_loss);
        ASSERT_LT(trained_loss, 0.5); // must fit well
        ASSERT_GT(random_loss, 0.5);
        torch::save(model, "tmp/tictactoe_model.pt");

    }
    // test

    {
        TicTacToeModel model;
        TicTacToeModel rand_model;
        torch::load(model, "tmp/tictactoe_model.pt");
        TicTacToe game;
        mcts_model_self_play<TicTacToe>(game, model, rand_model, 256, 10, 0.1, 2 );

        TicTacToe game1;
        mcts_model_self_play<TicTacToe>(game1, rand_model, model, 256, 10, 0.1, 2 );

        float losses = compare_models<TicTacToe>(model, rand_model, 256, 0.5, 1);

        cout << "compare to random " << losses << endl;
        ASSERT_LT(losses, 0.5);

        float rnd_losses = compare_models<TicTacToe>(rand_model, model, 256, 0.5, 1);
        cout << "compare to trained " << rnd_losses << endl;

        ASSERT_GT(rnd_losses, losses);
    }
}