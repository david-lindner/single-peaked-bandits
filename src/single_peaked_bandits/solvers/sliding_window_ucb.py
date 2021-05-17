import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.base import BaseSolver


class SlidingWindowUCB(BaseSolver):
    def __init__(self, xi=0.6, tau=None):
        super().__init__("sliding_window_ucb")
        self.xi = xi
        self.tau = tau
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
        tau,
    ):
        for i in arms_to_update:
            t_tau = min(timestep, tau)
            count_arm_in_last_tau = sum(
                [1 if history_arms[j] == i else 0 for j in range(-t_tau, 0)]
            )
            if count_arm_in_last_tau < 1e-10:
                optimistic_bounds[i] = float("inf")
            else:
                empirical_average = (
                    sum(
                        [
                            history_values[j] if history_arms[j] == i else 0
                            for j in range(-t_tau, 0)
                        ]
                    )
                    / count_arm_in_last_tau
                )
                exploration_bonus = B * np.sqrt(xi * np.log(t_tau) / count_arm_in_last_tau)
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

        if self.tau is None:
            tau = int(4 * np.sqrt(T * np.log(T)))
        else:
            tau = self.tau

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
                tau,
            )
            i_star = np.argmax(optimistic_bounds)
            policy[i_star] += 1
            arms_to_update = [i_star]
            timestep += 1
            history_arms.append(i_star)
            history_values.append(bandit.arms[i_star](policy[i_star]))

        return policy
