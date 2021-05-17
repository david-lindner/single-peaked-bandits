from single_peaked_bandits.solvers.base import BaseSolver


class GreedySolver(BaseSolver):
    def __init__(self):
        super().__init__("greedy")

    def solve(self, bandit, T):
        arms = bandit.arms
        n = len(arms)
        timestep = n
        policy = [1] * n

        while timestep < T:
            max_val = 0
            max_i = 0
            for i, f in enumerate(arms):
                val = f(policy[i])
                if val > max_val:
                    max_val = val
                    max_i = i
            policy[max_i] += 1
            timestep += 1

        return policy
