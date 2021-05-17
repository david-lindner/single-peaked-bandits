import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from single_peaked_bandits.constants import RESULTS_FOLDER, PLOTS_FOLDER, PICKLE_FOLDER
from single_peaked_bandits.experiments import ALL_BANDITS
from single_peaked_bandits.helpers import get_reward_for_policy

sns.set_context("paper", font_scale=1.8, rc={"lines.linewidth": 3})
sns.set_style("white")
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rc("text", usetex=True)

SOLVER_NAMES_FOR_PLOT = {
    "optimistic": "SPO (ours)",
    "increasing": "Increasing",
    "one_step_optimistic": "One-step-optimistic",
    "greedy": "Greedy",
    "exp3": "EXP3",
    "sliding_window_ucb": "SW-UCB",
    "discounted_ucb": "D-UCB",
    "Rexp3": "R-EXP3",
    "optimal": "Optimal Policy",
    "ucb": "UCB",
}
SOLVER_NAMES_FOR_PLOT = defaultdict(lambda: "", SOLVER_NAMES_FOR_PLOT)

# REWARD_CURVE_COLORS = ["cyan", "magenta", "red", "blue"]
REWARD_CURVE_COLORS = ["#f1a340", "#998ec3", "mediumseagreen", "navy"]
REWARD_CURVE_LINESTYLES = ["-", "--", "-.", ":"]

SOLVER_COLORS_FOR_PLOT = {
    "optimistic": "red",
    "increasing": "orange",
    "one_step_optimistic": "light purple",
    "greedy": "light green",
    "exp3": "mauve",
    "sliding_window_ucb": "light blue",
    "discounted_ucb": "mustard yellow",
    "optimal": "black",
    "ucb": "light blue",
    "Rexp3": "blue",
}
SOLVER_COLORS_FOR_PLOT = defaultdict(lambda: "blue", SOLVER_COLORS_FOR_PLOT)

SOLVER_LINESTYLES_FOR_PLOT = {
    "optimistic": "-",
    "exp3": "-.",
    "Rexp3": "-.",
    "sliding_window_ucb": ":",
    "discounted_ucb": ":",
}
SOLVER_LINESTYLES_FOR_PLOT = defaultdict(lambda: "--", SOLVER_LINESTYLES_FOR_PLOT)

SOLVER_ALPHAS_FOR_PLOT = {
    "optimistic": 1,
    "optimal": 1,
}
SOLVER_ALPHAS_FOR_PLOT = defaultdict(lambda: 0.9, SOLVER_ALPHAS_FOR_PLOT)

SOLVER_SIZES_FOR_PLOT = {
    "optimistic": 4,
    "optimal": 3,
}
SOLVER_SIZES_FOR_PLOT = defaultdict(lambda: 3, SOLVER_SIZES_FOR_PLOT)

SOLVER_ZORDERS_FOR_PLOT = {
    "optimistic": 2,
    "optimal": 1,
}
SOLVER_ZORDERS_FOR_PLOT = defaultdict(lambda: 1, SOLVER_ZORDERS_FOR_PLOT)

SOLVERS_TO_PLOT = {
    "optimistic",
    "increasing",
    "one_step_optimistic",
    "greedy",
    "exp3",
    "sliding_window_ucb",
    "discounted_ucb",
    "Rexp3",
    "optimal",
    "ucb",
}


def plot_rewards(ax, bandit, log_x_axis=False, title=True, xmax=None):
    colors = sns.color_palette(REWARD_CURVE_COLORS)
    i = 0
    t = np.arange(1, bandit.Tmax + 1, dtype=np.float)
    for f in bandit.arms:
        vf = np.vectorize(f, otypes=[np.float])
        ax.plot(
            t,
            vf(t),
            color=colors[i % len(colors)],
            label="Arm {}".format(i + 1),
            linestyle=REWARD_CURVE_LINESTYLES[i % len(colors)],
        )
        i += 1
    if log_x_axis:
        ax.set_xscale("log")
    ax.set_xlabel("Number of Pulls")
    ax.set_ylabel("Reward")
    if title:
        ax.set_title(bandit.name.replace("_", "\_"))
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.legend(frameon=True, prop={"size": 12})


def plot_value(
    ax,
    T_arrays,
    solver_names,
    values_arrays,
    bandit_name,
    log_x_axis=False,
    title=True,
    ylabel=None,
    xmax=None,
    ymax=None,
    legend=True,
):
    assert len(T_arrays) == len(solver_names)
    assert len(solver_names) == len(values_arrays)
    i = 0
    max_value = -float("inf")
    for solver_name, T, values in zip(solver_names, T_arrays, values_arrays):
        if solver_name in SOLVERS_TO_PLOT:
            assert len(values) > 0
            if len(values) == 1:
                value_mean = values[0]
            else:
                value_arr = np.stack(values)
                value_mean = np.mean(value_arr, axis=0)

            color = sns.xkcd_rgb[SOLVER_COLORS_FOR_PLOT[solver_name]]
            linestyle = SOLVER_LINESTYLES_FOR_PLOT[solver_name]
            label = SOLVER_NAMES_FOR_PLOT[solver_name]
            alpha = SOLVER_ALPHAS_FOR_PLOT[solver_name]
            size = SOLVER_SIZES_FOR_PLOT[solver_name]
            zorder = SOLVER_ZORDERS_FOR_PLOT[solver_name]
            ax.plot(
                T,
                value_mean,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=size,
                zorder=zorder,
            )

            if value_mean.shape[0] > 100:
                max_solver_value = value_mean[10:].max()
            else:
                max_solver_value = value_mean[:].max()
            if max_solver_value > max_value:
                max_value = max_solver_value

    if log_x_axis:
        ax.set_xscale("log")
    ax.set_xlabel("Time-Horizon")
    ax.set_ylabel(ylabel)

    # ymin = min(value_mean.min(), 0)
    # ymax = max_value * 1.05
    #
    # if ymax > 0:
    #     ax.set_ylim(ymin, ymax)

    if xmax is not None:
        ax.set_xlim(0, xmax)
    if ymax is not None:
        ax.set_ylim(0, ymax)


    if title:
        ax.set_title(bandit_name.replace("_", "\_"))
    if legend:
        ax.legend(
            frameon=True,
            ncol=1,
            prop={"size": 12},
            fancybox=True,
            framealpha=0.3,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )


def make_plots(
    separate_plots=False,
    title=True,
    log_x_axis=False,
    xmax=None,
    ymax=None,
    ylabel_performance=None,
    compute_cumulative=False,
):
    pickle_path = os.fsencode(os.path.join(RESULTS_FOLDER, PICKLE_FOLDER))
    plot_path = os.path.join(RESULTS_FOLDER, PLOTS_FOLDER)
    os.makedirs(plot_path, exist_ok=True)

    # Aggregate results
    results = dict()
    optimal_results = dict()
    for file in sorted(os.listdir(pickle_path)):
        filename = os.fsdecode(file)
        if filename.endswith(".p"):
            path = os.path.join(RESULTS_FOLDER, PICKLE_FOLDER, filename)
            print(filename)
            with open(path, "rb") as pickle_file:
                (
                    bandit_name,
                    solver_name,
                    T_list,
                    policy_list,
                    cumulative_reward_list,
                ) = pickle.load(pickle_file)

                if solver_name == "optimal":
                    n_seeds = min(
                        [i for i, p in enumerate(policy_list) if len(p) == 0]
                        + [len(policy_list)]
                    )
                    policy_list = policy_list[:n_seeds]
                    cumulative_reward_list = cumulative_reward_list[:n_seeds]
                    assert (
                        len(policy_list) == 1 and len(cumulative_reward_list) == 1
                    ), "optimal should only be run with a single seed"
                    if bandit_name not in optimal_results:
                        optimal_results[bandit_name] = dict()
                    for i, T in enumerate(T_list):
                        optimal_results[bandit_name][T] = (
                            policy_list[0][i],
                            cumulative_reward_list[0][i],
                        )
                else:
                    if bandit_name not in results:
                        results[bandit_name] = dict()
                    if solver_name not in results[bandit_name]:
                        results[bandit_name][solver_name] = (
                            T_list,
                            policy_list,
                            cumulative_reward_list,
                        )
    print("Found optimal policies for:", ", ".join(optimal_results.keys()))

    # Make plots
    for bandit_name, bandit_results in results.items():
        print("Making plots for", bandit_name)

        bandit = ALL_BANDITS[bandit_name]
        make_policy_plot = len(bandit.arms) == 2

        height, width = plt.figaspect(1.3)
        if not separate_plots:
            if make_policy_plot:
                fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), dpi=400)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(2 * width, height), dpi=400)
        if separate_plots:
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
        else:
            ax = axes[0]

        # Plot arms
        plot_rewards(ax, bandit, log_x_axis=log_x_axis, title=title, xmax=xmax)
        plt.tight_layout()

        if separate_plots:
            plt.savefig(
                os.path.join(plot_path, f"{bandit_name}_arms.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
        else:
            ax = axes[1]

        bandit_name_noise_free = bandit_name.split("_gaussian_noise")[0].split("_std")[
            0
        ]

        # Plot regret / return
        plot_regret = True
        solver_names = list(bandit_results.keys())

        # make sure ours is last in legend
        if "optimistic" in solver_names:
            i_ours = solver_names.index("optimistic")
            i_ours = solver_names.pop(i_ours)
            solver_names.append("optimistic")

        return_arrays, regret_arrays, first_arm_count_arrays, T_arrays = [], [], [], []
        for solver_name in solver_names:
            T_list, policy_list, cumulative_reward_list = bandit_results[solver_name]
            T_array = np.array(T_list)
            T_arrays.append(T_array)
            n_seeds = min(
                [i for i, p in enumerate(policy_list) if len(p) == 0]
                + [len(policy_list)]
            )
            print(f"\t{solver_name} {n_seeds} seeds")

            return_array, regret_array, first_arm_count_array = [], [], []
            for i in range(n_seeds):
                first_arm_count_array.append(np.array(policy_list[i])[:, 0])

                if compute_cumulative:
                    return_array.append(np.array(cumulative_reward_list[i]))
                else:
                    return_array.append(np.array(cumulative_reward_list[i]) / T_array)
                if plot_regret:
                    regret_list = []
                    for T, reward in zip(T_list, cumulative_reward_list[i]):
                        if (
                            bandit_name_noise_free in optimal_results
                            and T in optimal_results[bandit_name_noise_free]
                        ):
                            if compute_cumulative:
                                regret = (
                                    optimal_results[bandit_name_noise_free][T][1]
                                    - reward
                                )
                            else:
                                regret = (
                                    optimal_results[bandit_name_noise_free][T][1]
                                    - reward
                                ) / T
                            if regret < 0:
                                print(f"Warning: Negative regret {regret}")
                                regret = 0
                            regret_list.append(regret)
                        else:
                            print(
                                "No optimal result found for:", bandit_name_noise_free
                            )
                            plot_regret = False
                            break
                    regret_array.append(np.array(regret_list))

            first_arm_count_arrays.append(first_arm_count_array)
            return_arrays.append(return_array)
            regret_arrays.append(regret_array)

        result_arrays = regret_arrays if plot_regret else return_arrays

        if ylabel_performance is not None:
            ylabel = ylabel_performance
        else:
            ylabel = "Per-Step Regret" if plot_regret else "Per-Step Return"
        plot_value(
            ax,
            T_arrays,
            solver_names,
            result_arrays,
            bandit_name,
            log_x_axis=log_x_axis,
            title=title,
            ylabel=ylabel,
            xmax=xmax,
            ymax=ymax,
            legend=not separate_plots and not make_policy_plot,
        )
        plt.tight_layout()

        if separate_plots:
            plt.savefig(
                os.path.join(plot_path, f"{bandit_name}_regret.pdf"),
                bbox_inches="tight",
            )

        if make_policy_plot:
            if separate_plots:
                fig = plt.figure(figsize=(width, height))
                ax = plt.gca()
            else:
                ax = axes[2]

            if bandit_name_noise_free in optimal_results:
                T_list = sorted(optimal_results[bandit_name_noise_free].keys())
                T_list = [T for T in T_list if T < np.max(T_arrays)]
                optimal_policy = [
                    optimal_results[bandit_name_noise_free][T][0] for T in T_list
                ]
                T_arrays.append(np.array(T_list))
                first_arm_count_arrays.append([np.array(optimal_policy)[:, 0]])
                solver_names.append("optimal")

            plot_value(
                ax,
                T_arrays,
                solver_names,
                first_arm_count_arrays,
                bandit_name,
                log_x_axis=log_x_axis,
                title=title,
                ylabel="\# Pulls Arm 1",
                xmax=xmax,
                legend=not separate_plots,
            )
            plt.tight_layout()

            if separate_plots:
                plt.savefig(
                    os.path.join(plot_path, f"{bandit_name}_policy.pdf"),
                    bbox_inches="tight",
                )

        if not separate_plots:
            plt.subplots_adjust(hspace=0.25, wspace=0.40)
            plt.savefig(
                os.path.join(plot_path, f"{bandit_name}.pdf"), bbox_inches="tight"
            )
        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--separate_plots", action="store_true")
    parser.add_argument("--title", action="store_true")
    parser.add_argument("--log_x_axis", action="store_true")
    parser.add_argument("--xmax", type=int, default=None)
    parser.add_argument("--ymax", type=int, default=None)
    parser.add_argument("--ylabel_performance", type=str, default=None)
    parser.add_argument("--compute_cumulative", action="store_true")
    args = parser.parse_args()

    make_plots(
        separate_plots=args.separate_plots,
        title=args.title,
        log_x_axis=args.log_x_axis,
        xmax=args.xmax,
        ymax=args.ymax,
        ylabel_performance=args.ylabel_performance,
        compute_cumulative=args.compute_cumulative,
    )


if __name__ == "__main__":
    main()
