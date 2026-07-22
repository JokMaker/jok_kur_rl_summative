"""
main.py - single entry point required by the assignment spec.

Usage (after `uv sync`):
    uv run main.py                     # runs the best trained agent, rendered
    uv run main.py --algo dqn          # force a specific algorithm's best model
    uv run main.py --episodes 3        # number of episodes to simulate
    uv run main.py --no-render         # headless run (prints verbose logs only)

This is what a marker (or you, recording the demo video) runs to see the
agent in action: it loads the best saved model, resets the environment,
and steps through episodes while rendering the OpenGL "financial village"
and printing verbose terminal output of every decision.
"""
from __future__ import annotations

import argparse
import os
import sys

from environment.custom_env import MicroloanAllocationEnv, BASE_CAPITAL, ACTIONS

DQN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dqn")
PG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pg")


def load_best_agent(algo: str):
    """Returns a predict_fn(obs) -> action for the requested/best algorithm."""
    if algo == "dqn":
        from stable_baselines3 import DQN
        path = os.path.join(DQN_DIR, "best_model.zip")
        model = DQN.load(path)
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    if algo == "ppo":
        from stable_baselines3 import PPO
        path = os.path.join(PG_DIR, "best_ppo.zip")
        model = PPO.load(path)
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    if algo == "a2c":
        from stable_baselines3 import A2C
        path = os.path.join(PG_DIR, "best_a2c.zip")
        model = A2C.load(path)
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    if algo == "reinforce":
        import torch
        from training.pg_training import ReinforcePolicy
        path = os.path.join(PG_DIR, "best_reinforce.pt")
        env = MicroloanAllocationEnv()
        policy = ReinforcePolicy(env.observation_space.shape[0], env.action_space.n)
        policy.load_state_dict(torch.load(path))
        policy.eval()

        def predict_fn(obs):
            action, _ = policy.act(obs, deterministic=True)
            return action
        return predict_fn

    raise ValueError(f"Unknown algo: {algo}")


def run(algo: str, episodes: int, render: bool):
    render_mode = "human" if render else None
    predict_fn = load_best_agent(algo)

    print(f"\n{'='*70}")
    print(f"  Microloan Allocation Agent  |  policy = {algo.upper()}")
    print(f"  Mission: keep a fintech savings-group fund solvent and growing")
    print(f"  while approving/rejecting member loan requests each cycle.")
    print(f"{'='*70}\n")

    for ep in range(1, episodes + 1):
        env = MicroloanAllocationEnv(render_mode=render_mode, seed=1000 + ep)
        obs, info = env.reset()
        done = False
        step_count = 0
        print(f"--- Episode {ep} | starting fund capital = {BASE_CAPITAL:,.2f} ---")

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            print(
                f"  cycle={info['cycle']:>4} | action={ACTIONS[action]:<17} | "
                f"outcome={str(info['outcome']):<10} | reward={reward:+7.2f} | "
                f"fund_capital={info['fund_capital']:>12,.2f}"
            )
            done = terminated or truncated

        outcome = (
            "SUCCESS (fund reached sustainability target)"
            if info["fund_capital"] >= BASE_CAPITAL * 1.5
            else "FAILURE (fund insolvent)"
            if info["fund_capital"] <= BASE_CAPITAL * 0.15
            else "TIMEOUT (max cycles reached)"
        )
        print(f"--- Episode {ep} ended after {step_count} cycles: {outcome} "
              f"| final fund = {info['fund_capital']:,.2f} ---\n")
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Run the best-trained microloan agent.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn",
                         help="Which algorithm's best saved model to run (default: dqn).")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true", help="Disable the OpenGL window.")
    args = parser.parse_args()

    try:
        run(args.algo, args.episodes, render=not args.no_render)
    except FileNotFoundError as e:
        print(f"\n[error] Could not find a saved model for '{args.algo}': {e}")
        print("Train it first, e.g.:")
        print(f"  uv run python -m training.dqn_training      # for --algo dqn")
        print(f"  uv run python -m training.pg_training --algo {args.algo}")
        sys.exit(1)


if __name__ == "__main__":
    main()
