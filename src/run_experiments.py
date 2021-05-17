import time
import os
import pickle
import argparse
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np

from single_peaked_bandits.solvers import OptimalSolver
from single_peaked_bandits.helpers import (
    get_reward_for_policy,
)
from single_peaked_bandits.constants import RESULTS_FOLDER, PICKLE_FOLDER
from single_peaked_bandits.experiments import EXPERIMENTS

from make_plots import make_plots


def run_bandit_solver(job):
    bandit = job["bandit"]
    solver = job["solver"]
    randomize_bandit = job["randomize_bandit"]
    compute_regret = job["compute_regret"]
    T = job["T"]
    solver_instance = solver()

    t = time.time()

    if randomize_bandit:
        bandit._new_random_arms()

    policy = solver_instance.solve(bandit, T)
    cumulative_reward = get_reward_for_policy(bandit.noise_free_arms, T, policy)

    if compute_regret:
        optimal_solver = OptimalSolver()
        optimal_policy = optimal_solver.solve(bandit, T)
        optimal_reward = get_reward_for_policy(
            bandit.noise_free_arms, T, optimal_policy
        )
        single_peaked_bandits = optimal_reward - cumulative_reward
    else:
        single_peaked_bandits = None

    print(
        f"{solver_instance.name},  {bandit.name},  T: {T},  time: {time.time() - t},  "
        f"policy: {policy},  cumulative_reward: {cumulative_reward}"
    )

    return {
        "bandit": bandit.name,
        "solver": solver_instance.name,
        "T": T,
        "policy": policy,
        "cumulative_reward": cumulative_reward,
        "single_peaked_bandits": single_peaked_bandits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        help="Experiments to launch.",
        default="inc_dec_1",
    )
    parser.add_argument(
        "--n_jobs", type=int, help="Number of jobs to launch in parallel.", default=1
    )
    parser.add_argument(
        "--n_discretization",
        type=int,
        help="Discretization of the time-horzion.",
        default=100,
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        help="Number of random seeds to run per experiment.",
        default=30,
    )
    parser.add_argument("--optimal", action="store_true")
    parser.add_argument("--randomize_bandit", action="store_true")
    parser.add_argument("--compute_regret", action="store_true")
    args = parser.parse_args()

    pickle_path = os.path.join(RESULTS_FOLDER, PICKLE_FOLDER)
    os.makedirs(pickle_path, exist_ok=True)
    labels = args.labels.split(",")
    print("Running experiments:", ", ".join(labels))

    # Collect selected experiments
    jobs = []
    for experiment_label in labels:
        experiment_bandits, solvers = EXPERIMENTS[experiment_label]
        if args.optimal:
            solvers = [OptimalSolver]
        for bandit_n, bandit in enumerate(experiment_bandits):
            if args.optimal:
                T_array = np.arange(1, bandit.Tmax + 1)
            else:
                T_array = np.linspace(
                    2 * bandit.n + 1, bandit.Tmax, args.n_discretization, dtype=np.int
                )
            for solver in solvers:
                if bandit.stochastic or solver.stochastic:
                    n_runs = args.n_seeds
                else:
                    n_runs = 1
                for _ in range(n_runs):
                    for T in T_array:
                        jobs.append(
                            {
                                "bandit": bandit,
                                "solver": solver,
                                "T": T,
                                "randomize_bandit": args.randomize_bandit,
                                "compute_regret": args.compute_regret,
                            }
                        )

    # Run experiments
    start_time = time.time()
    if args.n_jobs == 1:
        results = []
        for job in jobs:
            result = run_bandit_solver(job)
            results.append(result)
    else:
        with mp.get_context("spawn").Pool(args.n_jobs) as p:
            results = p.map(run_bandit_solver, jobs, chunksize=1)

    # Aggregate results
    results_aggregated = dict()
    for res in results:
        bandit = res["bandit"]
        solver = res["solver"]
        T = res["T"]
        policy = res["policy"]
        cumulative_reward = res["cumulative_reward"]
        single_peaked_bandits = res["single_peaked_bandits"]

        if (bandit, solver) not in results_aggregated:
            results_aggregated[(bandit, solver)] = dict()
        if T not in results_aggregated[(bandit, solver)]:
            results_aggregated[(bandit, solver)][T] = []
        results_aggregated[(bandit, solver)][T].append(
            (policy, cumulative_reward, single_peaked_bandits)
        )

    # Write results
    for (bandit, solver), results in results_aggregated.items():
        T_list = []
        policy_list = [[] for _ in range(args.n_seeds)]
        cumulative_reward_list = [[] for _ in range(args.n_seeds)]
        for T in sorted(results.keys()):
            T_list.append(T)
            for i in range(len(results[T])):
                policy, cumulative_reward, single_peaked_bandits = results[T][i]
                policy_list[i].append(tuple(policy))
                if args.compute_regret:
                    cumulative_reward_list[i].append(single_peaked_bandits)
                else:
                    cumulative_reward_list[i].append(cumulative_reward)

        pickle_file = os.path.join(pickle_path, f"{bandit}_{solver}_result.p")
        with open(pickle_file, "wb") as f:
            pickle.dump(
                (
                    bandit,
                    solver,
                    tuple(T_list),
                    tuple([tuple(x) for x in policy_list]),
                    tuple([tuple(x) for x in cumulative_reward_list]),
                ),
                f,
            )

    make_plots()
    print("Done in", time.time() - start_time)


if __name__ == "__main__":
    main()
