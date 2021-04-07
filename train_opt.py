import os
import sys
import json

import optuna
import subprocess


def run_bin(trial, bin):
    params = {
        "train_learning_rate": trial.suggest_float("train_learning_rate", 1e-5, 1e-2, log=True),
        "train_l2_regularization": trial.suggest_float("train_l2_regularization", 1e-10, 1e-2, log=True),
        "train_replay_buffer": trial.suggest_categorical("train_replay_buffer", [2 ** i for i in range(8, 15)]),
        "train_epochs": trial.suggest_categorical("train_epochs", [2 ** i for i in range(0, 9)]),
        "train_batch_size": trial.suggest_categorical("train_batch_size", [2 ** i for i in range(3, 9)]),

        "simulation_cycles": trial.suggest_int(10000), # + inf
        "simulation_cycle_games": trial.suggest_categorical("simulation_cycle_games", [2 ** i for i in range(5, 12)]),
        "simulation_temperature": trial.suggest_float("simulation_temperature", 0.3, 1),

        "mcts_iterations": trial.suggest_categorical("mcts_iterations", [2 ** i for i in range(5, 10)]),
        "mcts_exploration": trial.suggest_float("mcts_exploration", 0.3, 3),

        "eval_size": 200,
        "eval_temperature": 0.1,
        "timeout": 300
    }

    try:
        output = subprocess.check_output([bin, '--config', json.dumps(params)])
        return float(output.split()[-1])
    except subprocess.CalledProcessError as er:
        raise optuna.TrialPruned()


def train_opt(bin):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: run_bin(trial, bin), n_trials=1000, n_jobs=8, show_progress_bar=True)


if __name__ == "__main__":
    bin = sys.argv[1]
    train_opt(bin)
