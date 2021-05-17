import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.base import BaseSolver


class DiscountedUCB(BaseSolver):
    def __init__(self, xi=0.6, gamma=None):
        super().__init__("discounted_ucb")
        self.xi = xi
        self.gamma = gamma
        self.B = 1  # bound on rewards

    def _update_optimistic_bound(
        self,
        bandit,
        policy,
        T,
        timestep,
        optimistic_bounds,
        arms_to_update,
        history_arms,
        history_values,
        B,
        xi,
        gamma,
    ):
        discounted_count = [None] * len(bandit.arms)
        sum_discounted_count = 0
        for i in range(len(bandit.arms)):
            discounted_count[i] = sum(
                [
                    gamma ** (timestep - j - 1) if history_arms[j] == i else 0
                    for j in range(timestep)
                ]
            )
            sum_discounted_count += discounted_count[i]

        for i in arms_to_update:
            if discounted_count[i] < 1e-10:
                optimistic_bounds[i] = float("inf")
            else:
                empirical_average = (
                    sum(
                        [
                            history_values[j] * gamma ** (timestep - j - 1)
                            if history_arms[j] == i
                            else 0
                            for j in range(timestep)
                        ]
                    )
                    / discounted_count[i]
                )
                exploration_bonus = B * np.sqrt(
                    xi * np.log(sum_discounted_count) / discounted_count[i]
                )
                optimistic_bounds[i] = empirical_average + exploration_bonus

    def solve(self, bandit, T):
        n_arms = len(bandit.arms)
        # pull every arm once
        n_init = 1
        timestep = n_init * n_arms
        policy = [n_init] * n_arms
        history_arms = list(range(n_arms))
        history_values = [bandit.arms[i](1) for i in history_arms]

        optimistic_bounds = np.zeros(n_arms)  # optimistic estimate of next step reward
        arms_to_update = range(n_arms)

        if self.gamma is None:
            gamma = 1 - 1 / (4*np.sqrt(T))
        else:
            gamma = self.gamma

        while timestep < T:
            self._update_optimistic_bound(
                bandit,
                policy,
                T,
                timestep,
                optimistic_bounds,
                arms_to_update,
                history_arms,
                history_values,
                self.B,
                self.xi,
                gamma,
            )
            i_star = np.argmax(optimistic_bounds)
            policy[i_star] += 1
            arms_to_update = [i_star]
            timestep += 1
            history_arms.append(i_star)
            history_values.append(bandit.arms[i_star](policy[i_star]))

        return policy


if __name__ == '__main__':
    from single_peaked_bandits.experiments import EXPERIMENT_INC_DEC_BANDITS
    from single_peaked_bandits.solvers import SlidingWindowUCB


    bandit = EXPERIMENT_INC_DEC_BANDITS[1]
    # solver = DiscountedUCB()
    solver = SlidingWindowUCB()
    policy = solver.solve(bandit, 100)
    print(policy)
