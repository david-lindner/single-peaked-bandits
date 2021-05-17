import numpy as np

from single_peaked_bandits.solvers.base import BaseSolver


class Rexp3(BaseSolver):
    stochastic = True

    def __init__(self, gamma=None, batch_size=None):
        super().__init__("Rexp3")
        self.gamma = gamma
        self.batch_size = batch_size

    def solve(self, bandit, T):
        """
        Implementation of https://pubsonline.informs.org/doi/pdf/10.1287/stsy.2019.0033
        """
        arms = bandit.arms
        K = len(arms)
        policy = [0] * K

        if self.batch_size is None:
            V = 2  # change budget
            batch_size = np.ceil((K * np.log(K)) ** (1 / 3) * (T / V) ** (2 / 3))
        else:
            batch_size = self.batch_size

        if self.gamma is None:
            gamma = min(1, np.sqrt(K * np.log(K) / ((np.e - 1) * batch_size)))
        else:
            gamma = self.gamma

        for t in range(T):
            if t % batch_size == 0:
                w = np.ones(K)
            p = (1 - gamma) * w / w.sum() + gamma / K
            i = np.random.choice(range(K), p=p)
            x_i = arms[i](policy[i])
            policy[i] += 1
            w[i] *= np.exp(gamma * (x_i / p[i]) / K)

        return policy
