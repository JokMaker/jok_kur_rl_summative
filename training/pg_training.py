"""
Policy-gradient training on MicroloanAllocationEnv: REINFORCE (custom
PyTorch implementation, since Stable-Baselines3 does not ship REINFORCE),
PPO, and A2C (both via Stable-Baselines3). Each algorithm gets a 10-run
hyperparameter sweep logged to logs/<ALGO>_runs.csv, and the best model
per algorithm is saved to models/pg/.

Usage:
    uv run python -m training.pg_training --algo reinforce --timesteps 40000
    uv run python -m training.pg_training --algo ppo --timesteps 40000
    uv run python -m training.pg_training --algo a2c --timesteps 40000
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import MicroloanAllocationEnv
from training.experiment_logger import log_run
from training.eval_utils import evaluate_policy_fn

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg"
)
os.makedirs(MODELS_DIR, exist_ok=True)


# ======================================================================
# REINFORCE (custom implementation - not available in SB3)
# ======================================================================

class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits

    def act(self, obs, deterministic=False):
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits).item()
            return action, None
        action = dist.sample()
        return action.item(), dist.log_prob(action)


REINFORCE_GRID = [
    dict(learning_rate=1e-2, gamma=0.99, entropy_coef=0.00),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.00),
    dict(learning_rate=1e-4, gamma=0.99, entropy_coef=0.00),
    dict(learning_rate=1e-3, gamma=0.95, entropy_coef=0.00),
    dict(learning_rate=1e-3, gamma=0.999, entropy_coef=0.00),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.01),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.05),
    dict(learning_rate=5e-3, gamma=0.97, entropy_coef=0.01),
    dict(learning_rate=5e-4, gamma=0.99, entropy_coef=0.02),
    dict(learning_rate=1e-3, gamma=0.90, entropy_coef=0.01),
]


def train_reinforce(run_id: int, hp: dict, total_timesteps: int):
    env = MicroloanAllocationEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = ReinforcePolicy(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=hp["learning_rate"])

    steps_done = 0
    episode_entropies = []

    while steps_done < total_timesteps:
        obs, _ = env.reset()
        log_probs, rewards, entropies = [], [], []
        done = False
        while not done:
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            steps_done += 1
            if steps_done >= total_timesteps:
                break

        # discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + hp["gamma"] * G
            returns.insert(0, G)
        returns = torch.as_tensor(returns, dtype=torch.float32)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        loss = -(log_probs_t * returns).sum() - hp["entropy_coef"] * entropies_t.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_entropies.append(entropies_t.mean().item())

    def predict_fn(o):
        action, _ = policy.act(o, deterministic=True)
        return action

    metrics = evaluate_policy_fn(predict_fn, n_episodes=20)
    metrics["mean_policy_entropy"] = float(np.mean(episode_entropies[-20:])) if episode_entropies else 0.0
    log_run("REINFORCE", run_id, hp, metrics)

    model_path = os.path.join(MODELS_DIR, f"reinforce_run{run_id}.pt")
    torch.save(policy.state_dict(), model_path)
    return metrics, model_path


# ======================================================================
# PPO (Stable-Baselines3)
# ======================================================================

PPO_GRID = [
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.0, n_steps=2048, clip_range=0.2),
    dict(learning_rate=1e-4, gamma=0.99, ent_coef=0.0, n_steps=2048, clip_range=0.2),
    dict(learning_rate=1e-3, gamma=0.99, ent_coef=0.0, n_steps=2048, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.95, ent_coef=0.0, n_steps=2048, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.999, ent_coef=0.0, n_steps=2048, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.01, n_steps=2048, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.05, n_steps=2048, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.0, n_steps=512, clip_range=0.2),
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.0, n_steps=2048, clip_range=0.1),
    dict(learning_rate=3e-4, gamma=0.99, ent_coef=0.0, n_steps=2048, clip_range=0.3),
]


def train_ppo(run_id: int, hp: dict, total_timesteps: int):
    env = Monitor(MicroloanAllocationEnv())
    model = PPO(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], gamma=hp["gamma"],
        ent_coef=hp["ent_coef"], n_steps=hp["n_steps"], clip_range=hp["clip_range"],
        verbose=0, tensorboard_log=os.path.join(MODELS_DIR, "tb_logs_ppo"),
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"ppo_run{run_id}")

    def predict_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    metrics = evaluate_policy_fn(predict_fn, n_episodes=20)
    log_run("PPO", run_id, hp, metrics)
    model_path = os.path.join(MODELS_DIR, f"ppo_run{run_id}.zip")
    model.save(model_path)
    return metrics, model_path


# ======================================================================
# A2C (Stable-Baselines3)
# ======================================================================

A2C_GRID = [
    dict(learning_rate=7e-4, gamma=0.99, ent_coef=0.0, n_steps=5),
    dict(learning_rate=1e-4, gamma=0.99, ent_coef=0.0, n_steps=5),
    dict(learning_rate=1e-3, gamma=0.99, ent_coef=0.0, n_steps=5),
    dict(learning_rate=7e-4, gamma=0.95, ent_coef=0.0, n_steps=5),
    dict(learning_rate=7e-4, gamma=0.999, ent_coef=0.0, n_steps=5),
    dict(learning_rate=7e-4, gamma=0.99, ent_coef=0.01, n_steps=5),
    dict(learning_rate=7e-4, gamma=0.99, ent_coef=0.05, n_steps=5),
    dict(learning_rate=7e-4, gamma=0.99, ent_coef=0.0, n_steps=20),
    dict(learning_rate=7e-4, gamma=0.99, ent_coef=0.0, n_steps=64),
    dict(learning_rate=3e-4, gamma=0.97, ent_coef=0.02, n_steps=10),
]


def train_a2c(run_id: int, hp: dict, total_timesteps: int):
    env = Monitor(MicroloanAllocationEnv())
    model = A2C(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], gamma=hp["gamma"],
        ent_coef=hp["ent_coef"], n_steps=hp["n_steps"],
        verbose=0, tensorboard_log=os.path.join(MODELS_DIR, "tb_logs_a2c"),
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"a2c_run{run_id}")

    def predict_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    metrics = evaluate_policy_fn(predict_fn, n_episodes=20)
    log_run("A2C", run_id, hp, metrics)
    model_path = os.path.join(MODELS_DIR, f"a2c_run{run_id}.zip")
    model.save(model_path)
    return metrics, model_path


# ======================================================================

ALGO_TABLE = {
    "reinforce": (REINFORCE_GRID, train_reinforce, "reinforce_run{}.pt"),
    "ppo": (PPO_GRID, train_ppo, "ppo_run{}.zip"),
    "a2c": (A2C_GRID, train_a2c, "a2c_run{}.zip"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGO_TABLE.keys()), required=True)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    grid, train_fn, _ = ALGO_TABLE[args.algo]
    best_metrics, best_path = None, None

    for run_id, hp in enumerate(grid[: args.runs], start=1):
        print(f"\n=== {args.algo.upper()} run {run_id}/{args.runs}: {hp} ===")
        metrics, model_path = train_fn(run_id, hp, args.timesteps)
        print(f"    -> mean_reward={metrics['mean_reward']:.2f} "
              f"success_rate={metrics['success_rate']:.2f}")
        if best_metrics is None or metrics["mean_reward"] > best_metrics["mean_reward"]:
            best_metrics, best_path = metrics, model_path

    if best_path is not None:
        ext = os.path.splitext(best_path)[1]
        best_dst = os.path.join(MODELS_DIR, f"best_{args.algo}{ext}")
        shutil.copy(best_path, best_dst)
        print(f"\nBest {args.algo.upper()} model ({best_metrics['mean_reward']:.2f} "
              f"mean reward) saved to {best_dst}")


if __name__ == "__main__":
    main()
