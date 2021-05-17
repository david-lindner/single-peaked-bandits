from functools import partial

import numpy as np


class NoiseModel:
    def __init__(self, bound):
        self.bound = bound

    def sample(self):
        raise NotImplementedError()


class SymmetricBetaNoiseModel(NoiseModel):
    def __init__(self, bound, var):
        self.beta = (1 / var - 4) / 8
        super().__init__(bound)

    def sample(self):
        return np.random.beta(self.beta, self.beta) * (2 * self.bound) - self.bound

    def __eq__(self, other):
        return self.beta == other.beta and self.bound == other.bound


class GaussianNoiseModel(NoiseModel):
    def __init__(self, std, mean=0, bound=None):
        self.mean = mean
        self.std = std
        if bound is None:
            bound = 2 * std
        super().__init__(bound)

    def sample(self):
        return np.random.normal(self.mean, self.std)

    def __eq__(self, other):
        return self.var == other.var and self.bound == other.bound


class Bandit:
    def __init__(self, name, arms, Tmax, noise_model=None, increasing=False):
        self.name = name
        self.n = len(arms)
        self.Tmax = Tmax
        self.noise_model = noise_model
        self.increasing = increasing
        self.set_arms(arms)

    def set_arms(self, arms):
        self.noise_free_arms = arms
        if self.noise_model:
            self.arms = []
            for i in range(len(arms)):
                self.arms.append(partial(self._noisy_arm, self.noise_free_arms, i))
            self.stochastic = True
        else:
            self.arms = self.noise_free_arms
            self.stochastic = False

    def _noisy_arm(self, arms, i, t):
        return arms[i](t) + self.noise_model.sample()

    def __str__(self):
        return "(" + ", ".join([str(arm) for arm in self.arms]) + ") " + str(self.Tmax)

class ConstantRewardBandit(Bandit):
    def __init__(self, name, n_arms, Tmax, noise_model=None, seed=None):
        self.seed = seed
        self.n_arms = n_arms
        super().__init__(name, [], Tmax, noise_model=noise_model, increasing=False)
        self._new_random_arms()

    def _arm(self, value, i):
        return value

    def _new_random_arms(self):
        old_seed = np.random.randint(0, 100000)
        np.random.seed(self.seed)

        arms = []
        for _ in range(self.n_arms):
            r = np.random.random()
            arms.append(partial(self._arm, r))
        self.set_arms(arms)

        self.seed = np.random.randint(0, 100000)
        np.random.seed(old_seed)


class RecommenderSystemBandit(Bandit):
    def __init__(
        self,
        name,
        Tmax,
        value,
        novelty,
        novelty_decay_gamma,
        novelty_decay_c,
        randomize=False,
        noise_model=None,
    ):
        self.Tmax = Tmax
        arms = self._get_arms(value, novelty, novelty_decay_gamma, novelty_decay_c)
        super().__init__(name, arms, Tmax, noise_model=noise_model, increasing=False)
        self.randomize = randomize
        if randomize:
            self.stochastic = True

    def _get_arms(self, value, novelty, novelty_decay_gamma, novelty_decay_c):
        assert len(value) == len(novelty)
        assert len(novelty) == len(novelty_decay_gamma)
        assert len(novelty_decay_gamma) == len(novelty_decay_c)
        arm_values = []
        for v, n, ndg, ndc in zip(value, novelty, novelty_decay_gamma, novelty_decay_c):
            arm_values.append(self._get_single_arm_values(v, n, ndg, ndc))
        arms = []
        for values in arm_values:
            arms.append(partial(self._values_arm, values))
        return arms

    def _get_single_arm_values(self, v, n, ndg, ndc):
        values = np.zeros(self.Tmax)
        for i in range(1, self.Tmax):
            values[i] = values[i - 1] + n * ndg ** i - ndc * (values[i - 1] - v)

        # linearly rescale if not already in [0,1]
        min_v, max_v = np.min(values), np.max(values)
        values -= min(min_v, 0)
        values /= max(max_v, 1)

        return values

    def _values_arm(self, values, i):
        return values[int(i) - 1]

    def _new_random_arms(self, seed=None):
        value, novelty, novelty_decay_gamma, novelty_decay_c = [], [], [], []
        if seed is not None:
            old_seed = np.random.randint(0, 100000)
            np.random.seed(seed)
        for i in range(self.n):
            value.append(np.random.choice([0, 0.5], p=[0.2, 0.8]) * np.random.random())
            novelty.append(0.5 * np.random.random())
            novelty_decay_gamma.append(0.9 + 0.1 * np.random.random())
            novelty_decay_c.append(0.2 * np.random.random())
        if seed is not None:
            np.random.seed(old_seed)
        self.set_arms(
            self._get_arms(value, novelty, novelty_decay_gamma, novelty_decay_c)
        )

    @classmethod
    def get_random(cls, name, Tmax, n_arms, noise_model=None, seed=None):
        obj = cls(
            name,
            Tmax,
            [0] * n_arms,
            [0] * n_arms,
            [0] * n_arms,
            [0] * n_arms,
            noise_model=noise_model,
            randomize=True,
        )
        obj._new_random_arms(seed=seed)
        return obj


class DataBandit(Bandit):
    def __init__(self, name, files, Tmax, noise_model=None, inject_noise=False):
        arm_data = []
        for f in files:
            rewards = np.load(f)
            arm_data.append(rewards)
            if Tmax is None:
                Tmax = rewards.shape[0]

        arms = []
        for rewards in arm_data:
            arms.append(partial(self._data_arm, rewards))

        super().__init__(name, arms, Tmax, noise_model=noise_model, increasing=False)
        # do not add noise
        if not inject_noise:
            self.arms = arms
            self.stochastic = False

    def _data_arm(self, rewards, i):
        if i > len(rewards):
            return 0
        return rewards[int(i) - 1]


if __name__ == "__main__":
    from make_plots import plot_rewards
    import matplotlib.pyplot as plt

    plt.figure()
    bandit = RecommenderSystemBandit.get_random("test", 500, 10)
    plot_rewards(plt.gca(), bandit)
    plt.show()
