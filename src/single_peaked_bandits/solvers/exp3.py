import numpy as np

from single_peaked_bandits.solvers.base import BaseSolver


class Exp3Solver(BaseSolver):
    stochastic = True

    def __init__(self, gamma=0.01):
        super().__init__("exp3")
        self.gamma = gamma

    def solve(self, bandit, T):
        """
        Implementation of http://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
        """
        arms = bandit.arms
        K = len(arms)
        w = np.ones(K)
        policy = [0] * K
        for t in range(T):
            p = (1 - self.gamma) * w / w.sum() + self.gamma / K
            i = np.random.choice(range(K), p=p)
            x_i = arms[i](policy[i])
            policy[i] += 1
            w[i] *= np.exp(self.gamma * (x_i / p[i]) / K)
        return policy


if __name__ == '__main__':
    from single_peaked_bandits.experiments import EXPERIMENT_INC_DEC_BANDITS

    bandit = EXPERIMENT_INC_DEC_BANDITS[0]
    T = 5000
    solver = Exp3Solver()
    policy = solver.solve(bandit, T)
    print(policy)
