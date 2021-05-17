from collections import defaultdict

import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.base import BaseSolver


class IncreasingSolver(BaseSolver):
    def __init__(self):
        super().__init__("increasing")

    def solve(self, bandit, T):
        arms = bandit.arms
        n = len(arms)
        S = set(range(n))
        phase = 0
        # pull every arm once
        timestep = n
        policy = [1] * n
        optimistic_bounds = {}
        pessimistic_bounds = {}
        cumulative_rewards = defaultdict(lambda: 0)
        while len(S) > 1 and timestep < T:
            phase += 1
            for arm in S:
                if timestep >= T:
                    break
                f = arms[arm]
                cumulative_rewards[arm] += f(phase)
                # pull arm
                timestep += 1
                policy[arm] += 1
                # update bounds
                optimistic_bounds[arm] = cumulative_rewards[arm]
                s_1 = T - phase
                if f(phase) - f(phase - 1) > 0:
                    s_1 = min(s_1, np.floor((1 - f(phase)) / (f(phase) - f(phase - 1))))
                optimistic_bounds[arm] += s_1 * f(phase)
                optimistic_bounds[arm] += (
                    (f(phase) - f(phase - 1)) * s_1 * (s_1 + 1) / 2
                )
                optimistic_bounds[arm] += max(T - phase - s_1, 0)
                pessimistic_bounds[arm] = cumulative_rewards[arm]
                pessimistic_bounds[arm] += f(phase) * (T - phase)
            else:  # no break
                for i in set(S):
                    for j in set(S):
                        if (
                            i != j
                            and pessimistic_bounds[j] > optimistic_bounds[i]
                            and i in S
                        ):
                            S.remove(i)
        if timestep < T:
            policy[S.pop()] += T - timestep
        return policy
