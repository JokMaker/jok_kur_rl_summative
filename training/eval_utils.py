"""Shared evaluation helper: runs a trained policy for N episodes and
returns summary statistics used both for hyperparameter tables and for
the generalization test in the report.
"""
from __future__ import annotations

import numpy as np

from environment.custom_env import MicroloanAllocationEnv, BASE_CAPITAL


def evaluate_policy_fn(predict_fn, n_episodes: int = 20, seed_offset: int = 1000):
    """
    predict_fn(obs) -> action (int). Works for both SB3 models
    (via model.predict) and the custom REINFORCE policy.
    """
    episode_rewards = []
    episode_lengths = []
    final_funds = []
    successes = 0

    for ep in range(n_episodes):
        env = MicroloanAllocationEnv(seed=seed_offset + ep)
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_funds.append(info["fund_capital"])
        if info["fund_capital"] >= BASE_CAPITAL * 1.5:
            successes += 1

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_final_fund": float(np.mean(final_funds)),
        "success_rate": successes / n_episodes,
    }
