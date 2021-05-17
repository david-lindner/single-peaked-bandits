import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.base import BaseSolver


class OneStepOptimisticSolver(BaseSolver):
    def __init__(self):
        super().__init__("one_step_optimistic")

    def _update_optimistic_bound(
        self, bandit, policy, T, timestep, optimistic_bounds, arms_to_update
    ):
        if bandit.noise_model is None:
            for i in arms_to_update:
                f = bandit.arms[i]
                n_i = policy[i]
                if f(n_i) - f(n_i - 1) > 0:
                    optimistic_bounds[i] = min(1, f(n_i) + f(n_i) - f(n_i - 1))
                else:
                    optimistic_bounds[i] = f(n_i)
        else:
            for i in arms_to_update:
                f = bandit.arms[i]
                n_i = policy[i]
                if f(n_i) - f(n_i - 1) > 0:
                    optimistic_bounds[i] = min(
                        1,
                        (f(n_i) + bandit.noise_model.bound)
                        + (
                            (f(n_i) + bandit.noise_model.bound)
                            - (f(n_i - 1) - bandit.noise_model.bound)
                        ),
                    )
                else:
                    optimistic_bounds[i] = f(n_i) + bandit.noise_model.bound

    def solve(self, bandit, T):
        n_arms = len(bandit.arms)
        n_init = 2
        # pull every arm twice
        timestep = n_init * n_arms
        policy = [n_init] * n_arms
        optimistic_bounds = np.zeros(n_arms)  # optimistic estimate of next step reward
        arms_to_update = range(n_arms)

        while timestep < T:
            self._update_optimistic_bound(
                bandit, policy, T, timestep, optimistic_bounds, arms_to_update
            )
            i_star = np.argmax(optimistic_bounds)
            policy[i_star] += 1
            arms_to_update = [i_star]
            timestep += 1

        return policy
