"""Basic sanity tests for the custom environment. Run with:
    uv run pytest tests/
"""
from environment.custom_env import MicroloanAllocationEnv, BASE_CAPITAL


def test_reset_returns_valid_obs():
    env = MicroloanAllocationEnv(seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["fund_capital"] == BASE_CAPITAL


def test_step_returns_valid_transition():
    env = MicroloanAllocationEnv(seed=0)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_reject_action_never_pays_out():
    env = MicroloanAllocationEnv(seed=1)
    obs, _ = env.reset()
    fund_before = env.fund_capital
    obs, reward, terminated, truncated, info = env.step(0)  # REJECT
    assert info["outcome"] == "rejected"
    assert env.fund_capital <= fund_before  # only cycle-growth bonus could differ, no loan payout


def test_episode_eventually_terminates():
    env = MicroloanAllocationEnv(seed=2)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done, "Episode should terminate or truncate within 500 steps"


def test_member_pool_persists_across_cycles():
    """The core fix: the same 40 Member objects must persist for an entire
    episode, so repeated decisions about one member compound rather than
    being redrawn fresh each cycle."""
    env = MicroloanAllocationEnv(seed=3)
    obs, _ = env.reset()
    member_ids_at_reset = [id(m) for m in env.members]
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    member_ids_later = [id(m) for m in env.members]
    assert member_ids_at_reset == member_ids_later, (
        "Member objects should be the same instances throughout the episode"
    )


def test_repayment_score_compounds_with_repeated_decisions():
    """Directly verifies the sequential-credit-assignment property: forcing
    repeated approvals on the same member should move their repayment_score
    away from its initial value based on actual outcomes, not reset it."""
    env = MicroloanAllocationEnv(seed=4)
    obs, _ = env.reset()
    target_idx = env.current_member_idx
    initial_score = env.members[target_idx].repayment_score

    for _ in range(10):
        env.current_member_idx = target_idx
        obs, reward, terminated, truncated, info = env.step(1)  # APPROVE_FULL
        if terminated or truncated:
            break

    final_score = env.members[target_idx].repayment_score
    assert final_score != initial_score, (
        "Repayment score should change based on accumulated repay/default history"
    )
