"""
Implementation of single-peaked optimism.
"""

try:
    import gurobipy

    gurobipy_available = True
    print("Using Gurobi")
except:
    import pulp
    from pulp.apis import PULP_CBC_CMD

    gurobipy_available = False
    print("Using CBC")


import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.base import BaseSolver


class OptimisticSolver(BaseSolver):
    def __init__(self, sigma=0):
        super().__init__("optimistic")
        self.gurobipy_env = None

    def _get_increasing_noise_bound_from_lp(self, L, U, T, t):
        n = len(L)
        assert len(U) == n
        D = T - t

        if gurobipy_available:
            if self.gurobipy_env is None:
                self.gurobipy_env = gurobipy.Env()
            with gurobipy.Model(env=self.gurobipy_env) as model:
                # the v's are the function values
                v = []
                for i in range(1, T + 1):
                    if i <= n:
                        lb = max(L[i - 1], 0)
                        ub = min(U[i - 1], 1)
                    else:
                        lb = 0
                        ub = 1

                    v_i = model.addVar(
                        name="v" + str(i), vtype=gurobipy.GRB.CONTINUOUS, lb=lb, ub=ub
                    )
                    v.append(v_i)
                    if i >= 3:
                        model.addConstr(v[-1] <= 2 * v[-2] - v[-3], "")
                    if i >= 2:
                        model.addConstr(v[-2] <= v[-1], "")

                # objective
                model.setObjective(sum(v[n + 1 :]), gurobipy.GRB.MAXIMIZE)

                model.setParam("OutputFlag", False)
                model.optimize()
                if model.status == gurobipy.GRB.OPTIMAL:
                    result = model.objVal
                else:
                    result = None
        else:
            model = pulp.LpProblem("noise lp", pulp.LpMaximize)

            # the v's are the function
            v = []
            for i in range(1, T + 1):
                if i <= n:
                    lb = max(L[i - 1], 0)
                    ub = min(U[i - 1], 1)
                else:
                    lb = 0
                    ub = 1

                v_i = pulp.LpVariable("v" + str(i), lowBound=lb, upBound=ub)
                v.append(v_i)
                if i >= 3:
                    model += v[-1] <= 2 * v[-2] - v[-3], ""
                if i >= 2:
                    model += v[-2] <= v[-1], ""

            # objective
            model += sum(v[n + 1 :])

            assert model.isMIP() == 0
            model.solve(PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[model.status] == "Optimal":
                result = pulp.value(model.objective)
            else:
                result = None

        return result

    def _update_optimistic_bound(
        self,
        bandit,
        T,
        t,
        optimistic_bounds,
        arms_to_update,
        values,
        decreasing_arms,
        L,
        U,
    ):
        if bandit.noise_model is None:
            for i in arms_to_update:
                if values[i][-1] - values[i][-2] > 0:
                    optimistic_bounds[i] = 0
                    for s in range(t + 1, T + 1):
                        optimistic_bounds[i] += min(
                            1, values[i][-1] + (values[i][-1] - values[i][-2]) * (s - t)
                        )
                else:
                    optimistic_bounds[i] = (T - t) * values[i][-1]
        else:
            for i in arms_to_update:
                if i not in decreasing_arms:
                    lp_result = self._get_increasing_noise_bound_from_lp(
                        L[i], U[i], T, t
                    )
                    if lp_result is None:
                        decreasing_arms.add(i)
                    else:
                        optimistic_bounds[i] = lp_result
                if i in decreasing_arms:
                    optimistic_bounds[i] = (T - t) * U[i][-1]

    def solve(self, bandit, T):
        n_arms = len(bandit.arms)
        # pull every arm log(T) times
        n_init = max(2, int(np.log(T)))
        timestep = n_init * n_arms
        policy = [n_init] * n_arms

        values = [[bandit.arms[i](t + 1) for t in range(n_init)] for i in range(n_arms)]
        if bandit.noise_model:
            eps = bandit.noise_model.bound
            L = [[v - eps for v in values[i]] for i in range(n_arms)]
            U = [[v + eps for v in values[i]] for i in range(n_arms)]
            decreasing_arms = set()
        else:
            L, U = None, None
            decreasing_arms = None

        optimistic_bounds = np.zeros(n_arms)
        arms_to_update = range(n_arms)

        while timestep < T:
            self._update_optimistic_bound(
                bandit,
                T,
                timestep,
                optimistic_bounds,
                arms_to_update,
                values,
                decreasing_arms,
                L,
                U,
            )
            i_star = np.argmax(optimistic_bounds)
            # print("timestep", timestep)
            # print("optimistic_bounds", optimistic_bounds)
            # print("i_star", i_star)
            policy[i_star] += 1
            value = bandit.arms[i_star](policy[i_star])
            values[i_star].append(value)
            if bandit.noise_model:
                L[i_star].append(value - eps)
                U[i_star].append(value + eps)

            arms_to_update = [i_star]
            timestep += 1

        del self.gurobipy_env
        self.gurobipy_env = None
        return policy
