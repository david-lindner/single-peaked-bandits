"""
Preprocessing codes based on https://github.com/lydiatliu/delayedimpact
"""
import itertools
import os

import numpy as np
from scipy.interpolate import interp1d

import fico


def get_raw_data_for_group(data_dir, group_str):
    all_cdfs, performance, totals = fico.get_FICO_data(data_dir=data_dir)
    cdf = all_cdfs[group_str].values
    repay = performance[group_str]
    scores = all_cdfs[group_str].index
    scores_list = scores.tolist()
    return scores_list, repay, cdf


def sample_group_population(N_sample, scores_list, cdf):
    inverted_cdf = interp1d(cdf, scores_list)
    rand = np.random.rand(N_sample)
    rand = np.clip(rand, np.min(inverted_cdf.x), np.max(inverted_cdf.x))
    scores_sample = np.array(inverted_cdf(rand))
    scores_sample = np.clip(scores_sample, 300, 850)
    return scores_sample


def get_reward(repay, scores_sample, util_repay, util_default):
    sorted_scores = sorted(scores_sample, reverse=True)
    repay_fn = interp1d(repay.index, repay.values)
    repay = repay_fn(sorted_scores)
    rewards = repay * util_repay + (1 - repay) * util_default
    rewards = np.clip(rewards / rewards.max(), 0, 1)
    return rewards


def get_score_change_reward(
    repay, scores_sample, score_change_repay, score_change_default
):
    sorted_scores = np.array(sorted(scores_sample, reverse=True))
    scr = np.clip(sorted_scores + score_change_repay, 300, 850) - sorted_scores
    scd = np.clip(sorted_scores + score_change_default, 300, 850) - sorted_scores
    rewards = get_reward(repay, scores_sample, scr, scd)
    return rewards


if __name__ == "__main__":
    fico_exp_folder = os.path.dirname(__file__)
    data_dir = os.path.join(fico_exp_folder, "data/")
    groups = ["Asian", "Black", "Hispanic", "White"]
    N_sample = 1000
    utility_repaid = 1
    utility_default = -4
    score_change_repay = 75
    score_change_default = -150

    for group_str in groups:
        scores_list, repay, cdf = get_raw_data_for_group(data_dir, group_str)
        scores_sample = sample_group_population(N_sample, scores_list, cdf)
        reward_utility = get_reward(
            repay, scores_sample, utility_repaid, utility_default
        )
        reward_score_change = get_score_change_reward(
            repay, scores_sample, score_change_repay, score_change_default
        )
        np.save(
            os.path.join(
                fico_exp_folder, "fico_reward_group_{}_utility.npy".format(group_str)
            ),
            reward_utility,
        )
        np.save(
            os.path.join(
                fico_exp_folder,
                "fico_reward_group_{}_score_change.npy".format(group_str),
            ),
            reward_score_change,
        )
