"""
MicroloanAllocationEnv
=======================

A custom Gymnasium environment modeling a fintech savings-group microloan
fund (inspired by the "savings groups" feature on Exuus's save-api-v3
platform). An agent (the loan officer) must decide how to respond to
sequential loan requests from group members in order to keep the fund
solvent and growing, while still serving the community it exists for.

WHY THIS IS A GENUINE SEQUENTIAL DECISION PROBLEM (not just per-state
classification): the group is a FIXED, PERSISTENT pool of members who are
cycled through repeatedly across an episode. A member's repayment_score
compounds based on the agent's own past decisions about that specific
member - approve them repeatedly and repay well, their score keeps
climbing and they qualify for bigger loans; approve someone who defaults
twice and their score craters, changing what the *optimal* action for them
becomes going forward. This means the agent must reason about the long-run
consequence of a decision, not just react to the current snapshot.

Action Space (Discrete(6)) - exhaustive, and each action maps directly
onto a real lending decision a human loan officer could take:
    0: REJECT               - decline the loan request entirely
    1: APPROVE_FULL         - approve full requested amount, standard terms
    2: APPROVE_PARTIAL      - approve 50% of requested amount, standard terms
    3: APPROVE_EXTENDED     - approve full amount, longer repayment window
    4: APPROVE_GRACE        - approve full amount, delayed first repayment
    5: APPROVE_STARTER      - approve a small fixed "starter" loan regardless
                              of the amount requested (financial-inclusion
                              action for thin-file / new members)

Observation Space (Box, 8-dim, all normalized to roughly [0, 1] or [-1, 1]):
    0: fund_liquidity_ratio      - available capital / total obligations
    1: member_repayment_score    - THIS member's repayment score [0,1],
                                   which evolves based on the agent's own
                                   past decisions about them
    2: member_relative_savings   - member savings balance vs. group average
    3: requested_amount_norm     - requested loan amount, normalized
    4: member_tenure_norm        - how long member has been in the group
    5: cycles_since_last_loan    - normalized time since member's last loan
    6: group_default_rate        - rolling default rate across the group
    7: season_indicator          - -1 (lean season) .. +1 (harvest season)

Rewards:
    + repayment_reward  * loan_size   on successful repayment
    - default_penalty   * loan_size   on default
    + growth_bonus                    each cycle the fund grows
    - insolvency_penalty               when liquidity ratio < critical threshold
    + inclusion_bonus                  for approving a thin-file/underserved
                                        member who then repays successfully

Start State:
    Fund initialized at BASE_CAPITAL. A FIXED population of N_MEMBERS members
    is sampled once at reset with randomized (but realistic-distribution)
    attributes, and that same population persists for the whole episode.

Terminal Conditions:
    - FAILURE:  fund_liquidity_ratio drops below CRITICAL_LIQUIDITY
    - SUCCESS:  fund reaches SUSTAINABILITY_TARGET * BASE_CAPITAL
    - TIMEOUT:  MAX_CYCLES loan-decision cycles reached
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----------------------------- Constants ------------------------------ #

BASE_CAPITAL = 100_000.0
N_MEMBERS = 40
MAX_CYCLES = 200

CRITICAL_LIQUIDITY = 0.15          # fund insolvency threshold
SUSTAINABILITY_TARGET = 1.5        # success if fund reaches 1.5x base capital

STARTER_LOAN_AMOUNT = 2_000.0
INTEREST_RATE = 0.18               # simple interest applied on repayment

# On default, the group recovers part of the loss from the member's own
# savings contribution - this is how real savings groups actually work
# (a defaulted loan is offset against the member's savings balance), and it
# ties the reward economics directly to an existing observation feature
# (member_relative_savings), rather than treating default as a total loss.
DEFAULT_RECOVERY_FRACTION = 0.55

REPAYMENT_REWARD_SCALE = 1.2
DEFAULT_PENALTY_SCALE = 2.0
GROWTH_BONUS = 0.5
INSOLVENCY_PENALTY = 50.0
INCLUSION_BONUS = 3.0
SUCCESS_BONUS = 100.0
FAILURE_PENALTY = -100.0

# How much a repayment/default nudges a member's score - this is what makes
# the environment sequential: today's decision changes tomorrow's state for
# THIS member, not just a fresh random draw.
REPAYMENT_SCORE_GAIN = 0.04
DEFAULT_SCORE_LOSS = 0.20

ACTIONS = {
    0: "REJECT",
    1: "APPROVE_FULL",
    2: "APPROVE_PARTIAL",
    3: "APPROVE_EXTENDED",
    4: "APPROVE_GRACE",
    5: "APPROVE_STARTER",
}


class Member:
    """
    A savings-group member with attributes that drive repayment probability.
    Members are created ONCE per episode (in reset) and persist across all
    cycles in that episode - their repayment_score updates over time based
    on the agent's own decisions about them.
    """

    def __init__(self, rng: np.random.Generator):
        self.repayment_score = float(np.clip(rng.normal(0.6, 0.18), 0.05, 0.99))
        self.savings_balance = float(max(0.0, rng.normal(5_000, 2_500)))
        self.tenure_cycles = int(rng.integers(0, 100))
        self.cycles_since_last_loan = int(rng.integers(0, 30))
        self.is_thin_file = self.tenure_cycles < 10 and self.savings_balance < 2_000
        # track outcomes for this member across the whole episode - lets us
        # analyze in the report whether the agent learns to "cut off" repeat
        # defaulters or "invest more" in reliable members over time
        self.loan_count = 0
        self.default_count = 0
        self.repaid_count = 0

    def request_amount(self, rng: np.random.Generator) -> float:
        base = 1_000 + self.savings_balance * rng.uniform(0.5, 2.0)
        return float(np.clip(base, 500, 20_000))


class MicroloanAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -3, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([3, 1, 3, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self._renderer = None  # lazily created (OpenGL) if render_mode == "human"

        # episode state (assigned in reset)
        self.fund_capital = BASE_CAPITAL
        self.members: list[Member] = []
        self.cycle = 0
        self.default_history: list[int] = []
        self.current_member_idx = 0
        self.current_request = 0.0
        self.season = 1.0
        self.last_action_name = None
        self.last_outcome = None  # for rendering: "repaid" / "defaulted" / "rejected" / None

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.fund_capital = BASE_CAPITAL
        # Fixed persistent pool for this episode - created once, reused every cycle.
        self.members = [Member(self._rng) for _ in range(N_MEMBERS)]
        self.cycle = 0
        self.default_history = []
        self.season = 1.0
        self.last_action_name = None
        self.last_outcome = None

        self._next_request()
        obs = self._get_obs()
        info = {"fund_capital": self.fund_capital}
        return obs, info

    # ------------------------------------------------------------------ #
    def _next_request(self):
        # A member is drawn from the SAME persistent pool each cycle - over
        # an episode, every member is revisited multiple times, and their
        # evolving repayment_score reflects the agent's cumulative decisions.
        self.current_member_idx = int(self._rng.integers(0, N_MEMBERS))
        member = self.members[self.current_member_idx]
        self.current_request = member.request_amount(self._rng)
        # season oscillates slowly to create structured (non-random-noise) shocks
        self.season = float(np.sin(self.cycle / 20.0))

    def _get_obs(self) -> np.ndarray:
        member = self.members[self.current_member_idx]
        avg_savings = np.mean([m.savings_balance for m in self.members]) + 1e-6
        liquidity_ratio = self.fund_capital / (BASE_CAPITAL * SUSTAINABILITY_TARGET)
        default_rate = (
            np.mean(self.default_history[-20:]) if self.default_history else 0.0
        )
        obs = np.array(
            [
                np.clip(liquidity_ratio, 0, 3),
                member.repayment_score,
                np.clip(member.savings_balance / avg_savings, -3, 3),
                np.clip(self.current_request / 20_000, 0, 1),
                np.clip(member.tenure_cycles / 100, 0, 1),
                np.clip(member.cycles_since_last_loan / 30, 0, 1),
                np.clip(default_rate, 0, 1),
                np.clip(self.season, -1, 1),
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        assert self.action_space.contains(action)
        member = self.members[self.current_member_idx]
        reward = 0.0
        terminated = False
        truncated = False
        self.last_action_name = ACTIONS[action]
        self.last_outcome = None

        if action == 0:  # REJECT
            reward -= 0.1  # small opportunity-cost penalty for over-caution
            self.last_outcome = "rejected"
        else:
            loan_size, term_multiplier, grace = self._action_terms(action, member)
            member.loan_count += 1

            # probability of repayment depends on member score, season, and
            # how aggressive the loan term is (grace/extended terms help)
            season_effect = 0.1 * self.season  # harvest season improves repayment odds
            term_effect = 0.05 * grace + 0.03 * (term_multiplier - 1)
            repay_prob = np.clip(
                member.repayment_score + season_effect + term_effect, 0.02, 0.98
            )

            repaid = self._rng.random() < repay_prob

            if repaid:
                interest_earned = loan_size * INTEREST_RATE * term_multiplier
                self.fund_capital += interest_earned
                reward += REPAYMENT_REWARD_SCALE * (loan_size / 10_000)
                if member.is_thin_file:
                    reward += INCLUSION_BONUS
                self.default_history.append(0)
                self.last_outcome = "repaid"
                member.repayment_score = float(
                    np.clip(member.repayment_score + REPAYMENT_SCORE_GAIN, 0.05, 0.99)
                )
                member.repaid_count += 1
            else:
                recovery = min(loan_size, member.savings_balance * DEFAULT_RECOVERY_FRACTION)
                net_loss = loan_size - recovery
                self.fund_capital -= net_loss
                reward -= DEFAULT_PENALTY_SCALE * (net_loss / 10_000)
                self.default_history.append(1)
                self.last_outcome = "defaulted"
                member.repayment_score = float(
                    np.clip(member.repayment_score - DEFAULT_SCORE_LOSS, 0.05, 0.99)
                )
                member.default_count += 1

            member.cycles_since_last_loan = 0

        member.tenure_cycles += 1
        for m in self.members:
            m.cycles_since_last_loan += 1

        # fund growth bonus (small, encourages sustainable growth each cycle)
        liquidity_ratio = self.fund_capital / (BASE_CAPITAL * SUSTAINABILITY_TARGET)
        if self.fund_capital > BASE_CAPITAL:
            reward += GROWTH_BONUS * liquidity_ratio

        # terminal conditions
        if liquidity_ratio < CRITICAL_LIQUIDITY:
            reward += FAILURE_PENALTY
            terminated = True
        elif self.fund_capital >= BASE_CAPITAL * SUSTAINABILITY_TARGET:
            reward += SUCCESS_BONUS
            terminated = True

        self.cycle += 1
        if self.cycle >= MAX_CYCLES:
            truncated = True

        if not (terminated or truncated):
            self._next_request()

        obs = self._get_obs()
        info = {
            "fund_capital": self.fund_capital,
            "cycle": self.cycle,
            "last_action": self.last_action_name,
            "outcome": self.last_outcome,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _action_terms(self, action: int, member: Member) -> tuple[float, float, float]:
        """Return (loan_size, term_multiplier, grace_flag) for a given approval action."""
        if action == 1:  # APPROVE_FULL
            return self.current_request, 1.0, 0.0
        if action == 2:  # APPROVE_PARTIAL
            return self.current_request * 0.5, 1.0, 0.0
        if action == 3:  # APPROVE_EXTENDED
            return self.current_request, 1.5, 0.0
        if action == 4:  # APPROVE_GRACE
            return self.current_request, 1.0, 1.0
        if action == 5:  # APPROVE_STARTER
            return STARTER_LOAN_AMOUNT, 1.0, 0.0
        raise ValueError(f"Unknown action {action}")

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode is None:
            return
        if self._renderer is None:
            from environment.rendering import OpenGLRenderer

            self._renderer = OpenGLRenderer(N_MEMBERS)
        self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    # quick smoke test
    env = MicroloanAllocationEnv(seed=42)
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        if term or trunc:
            break
    print("Smoke test finished. Total reward:", total_reward, "Info:", info)
