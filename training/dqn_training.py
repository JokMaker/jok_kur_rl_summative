"""
DQN training on MicroloanAllocationEnv, with a 10-run hyperparameter
sweep (learning_rate, gamma, buffer_size, exploration_fraction) whose
results are logged to logs/DQN_runs.csv for the report's hyperparameter
table, and the best model is saved to models/dqn/best_model.zip.
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import MicroloanAllocationEnv
from training.experiment_logger import log_run
from training.eval_utils import evaluate_policy_fn

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dqn"
)
os.makedirs(MODELS_DIR, exist_ok=True)

# 10 hyperparameter combinations spanning learning rate, gamma, buffer size,
# and exploration schedule - the four axes DQN behavior is most sensitive to.
HYPERPARAM_GRID = [
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=10_000, exploration_fraction=0.10),
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10_000, exploration_fraction=0.10),
    dict(learning_rate=5e-4, gamma=0.95, buffer_size=10_000, exploration_fraction=0.10),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=50_000, exploration_fraction=0.10),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=10_000, exploration_fraction=0.30),
    dict(learning_rate=1e-3, gamma=0.90, buffer_size=10_000, exploration_fraction=0.10),
    dict(learning_rate=2.5e-4, gamma=0.99, buffer_size=100_000, exploration_fraction=0.20),
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=10_000, exploration_fraction=0.50),
    dict(learning_rate=1e-4, gamma=0.995, buffer_size=50_000, exploration_fraction=0.20),
    dict(learning_rate=7e-4, gamma=0.97, buffer_size=25_000, exploration_fraction=0.15),
]


def make_env():
    return Monitor(MicroloanAllocationEnv())


def run_experiment(run_id: int, hyperparams: dict, total_timesteps: int):
    env = make_env()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        buffer_size=hyperparams["buffer_size"],
        exploration_fraction=hyperparams["exploration_fraction"],
        verbose=0,
        tensorboard_log=os.path.join(MODELS_DIR, "tb_logs"),
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"dqn_run{run_id}")

    def predict_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    metrics = evaluate_policy_fn(predict_fn, n_episodes=20)
    log_run("DQN", run_id, hyperparams, metrics)

    model_path = os.path.join(MODELS_DIR, f"dqn_run{run_id}.zip")
    model.save(model_path)
    return metrics, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40_000,
                         help="Timesteps per run. Increase for final results.")
    parser.add_argument("--runs", type=int, default=len(HYPERPARAM_GRID),
                         help="How many grid entries to run (max 10).")
    args = parser.parse_args()

    best_metrics = None
    best_path = None

    for run_id, hp in enumerate(HYPERPARAM_GRID[: args.runs], start=1):
        print(f"\n=== DQN run {run_id}/{args.runs}: {hp} ===")
        metrics, model_path = run_experiment(run_id, hp, args.timesteps)
        print(f"    -> mean_reward={metrics['mean_reward']:.2f} "
              f"success_rate={metrics['success_rate']:.2f}")
        if best_metrics is None or metrics["mean_reward"] > best_metrics["mean_reward"]:
            best_metrics = metrics
            best_path = model_path

    if best_path is not None:
        best_dst = os.path.join(MODELS_DIR, "best_model.zip")
        import shutil
        shutil.copy(best_path, best_dst)
        print(f"\nBest DQN model ({best_metrics['mean_reward']:.2f} mean reward) "
              f"saved to {best_dst}")


if __name__ == "__main__":
    main()
