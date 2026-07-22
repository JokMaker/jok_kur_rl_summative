# Microloan Allocation Agent — RL Summative

A mission-based reinforcement learning summative comparing **DQN**, **REINFORCE**,
**PPO**, and **A2C** on a custom environment modeling a fintech savings-group
microloan fund, inspired by the savings-groups feature I work on at Exuus.

## The Mission

A savings group has a shared fund and a **fixed pool of members** who
persist for the whole episode. Each cycle, one member (drawn from that same
pool) requests a loan. The agent (a loan officer) must decide how to respond
— approve, reject, or restructure — in order to keep the fund solvent and
growing, while still serving the community it exists for.

Because the member pool is persistent rather than freshly randomized every
cycle, a member's repayment score genuinely compounds based on the agent's
own past decisions about them: approve someone repeatedly and they repay
well, their score climbs and they qualify for larger loans; approve someone
who then defaults twice, their score craters and the optimal action for
them changes. This is what makes the problem a genuine sequential decision
problem rather than a per-state classification task — the agent has to
reason about the long-run consequence of a decision, not just react to a
snapshot.

Full environment specification (action space, observation space, reward
structure, start state, terminal conditions) is documented in
[`environment/custom_env.py`](environment/custom_env.py) and in the report.

## Project Structure

```
project_root/
├── pyproject.toml          # uv-managed dependencies
├── uv.lock
├── main.py                 # entry point — runs the best trained agent, rendered
├── environment/
│   ├── custom_env.py       # the Gymnasium environment
│   └── rendering.py        # OpenGL 3D "financial village" visualization
├── training/
│   ├── dqn_training.py     # DQN + 10-run hyperparameter sweep
│   ├── pg_training.py      # REINFORCE (custom) + PPO + A2C, each with sweeps
│   ├── eval_utils.py
│   └── experiment_logger.py
├── api/
│   └── serve.py            # FastAPI endpoint serving the trained policy
├── models/
│   ├── dqn/
│   └── pg/
├── logs/                    # CSV results from hyperparameter sweeps
├── assets/                  # plots and figures for the report
└── tests/
```

## Setup (uv only — no manual pip/venv steps required)

```bash
git clone https://github.com/JokMaker/jok_kur_rl_summative.git
cd jok_kur_rl_summative
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and builds the exact
dependency environment used to develop this project — no manual
`pip install` or venv creation needed.

## Training

Each script runs its full 10-run hyperparameter sweep and logs results to
`logs/<ALGO>_runs.csv`, saving the best model to `models/`.

```bash
uv run python -m training.dqn_training --timesteps 100000
uv run python -m training.pg_training --algo reinforce --timesteps 100000
uv run python -m training.pg_training --algo ppo --timesteps 100000
uv run python -m training.pg_training --algo a2c --timesteps 100000
```

Increase `--timesteps` for stronger final policies; decrease for a quick check.

## Running the Best Agent (`main.py`)

```bash
uv run main.py                     # runs the best DQN agent, rendered in OpenGL
uv run main.py --algo ppo          # run a different algorithm's best model
uv run main.py --episodes 5        # simulate more episodes
uv run main.py --no-render         # headless, terminal-only verbose output
```

This opens the 3D "financial village" window (each member is a building —
height = savings balance, color = repayment health) and prints every loan
decision, outcome, reward, and running fund balance to the terminal.

## Serving the Agent as an API

```bash
uv run uvicorn api.serve:app --reload
```

Then `POST /decide-loan` with a member's state to get back the agent's
decision as JSON — showing how the trained policy could be integrated into
a real lending product's backend.

## Tests

```bash
uv run pytest tests/
```

Covers basic environment sanity plus two tests specifically verifying the
persistent-member-pool design: that member objects persist across an
episode, and that a member's repayment score compounds with repeated
agent decisions rather than resetting each cycle.

## Author

Jok Maker Kur — Final-year Software Engineering student, African Leadership
University (ALU), specializing in Machine Learning.
