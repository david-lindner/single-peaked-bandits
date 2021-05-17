import itertools

import numpy as np

from single_peaked_bandits.solvers.base import BaseSolver
from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.bandits import ConstantRewardBandit


def get_optimal_arm_constant_reward_bandit(arms):
    values = [arm(1) for arm in arms]
    return np.argmax(values)

def optimal_policy_constant_reward_bandit(arms, T):
    n = len(arms)
    optimal_arm = get_optimal_arm_constant_reward_bandit(arms)
    policy = [0] * n
    policy[optimal_arm] = T
    return policy

def get_optimal_arm_increasing_reward_bandit(arms, T):
    n = len(arms)
    optimal_arm = 0
    optimal_reward = 0
    for candidate_arm in range(n):
        f = arms[candidate_arm]
        candidate_reward = cumulative_reward(f, T)
        if candidate_reward >= optimal_reward:
            optimal_arm = candidate_arm
            optimal_reward = candidate_reward
    return optimal_arm


def optimal_policy_increasing_reward_bandit(arms, T):
    n = len(arms)
    optimal_arm = get_optimal_arm_increasing_reward_bandit(arms, T)
    policy = [0] * n
    policy[optimal_arm] = T
    return policy


def successor(n, l):
    idx = [j for j in range(len(l)) if l[j] < l[0] - 1]
    if not idx:
        return False
    i = idx[0]  # smallest index i s.t. a_i < a_1 - 1
    l[1 : i + 1] = [l[i] + 1] * i
    l[0] = n - sum(l[1:])
    return True


def partitions(n, k):
    """ Get all partitions of integer n into k summands.

    Source: https://stackoverflow.com/a/14693916
    """
    if n == 1:
        yield [1]+[0]*(k-1)
    else:
        l = [0] * k
        l[0] = n
        yield list(l)
        while successor(n, l):
            yield list(l)

class OptimalSolver(BaseSolver):
    """Find optimal policy by bruteforce."""

    def __init__(self):
        super().__init__("optimal")

    def solve(self, bandit, T):
        arms = bandit.noise_free_arms
        if bandit.increasing:
            return optimal_policy_increasing_reward_bandit(arms, T)
        elif isinstance(bandit, ConstantRewardBandit):
            return optimal_policy_constant_reward_bandit(arms, T)

        n_arms = len(arms)

        best_policy = []
        best_return = -float("inf")

        for partition in partitions(T, n_arms):
            for part in itertools.permutations(partition):
                ret = sum([cumulative_reward(arm, t) for arm, t in zip(arms, part)])
                if ret > best_return:
                    best_policy = part
                    best_return = ret

        return best_policy
