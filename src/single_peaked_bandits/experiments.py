import os
import itertools
from functools import partial

import numpy as np

from single_peaked_bandits.bandits import (
    Bandit,
    DataBandit,
    RecommenderSystemBandit,
    SymmetricBetaNoiseModel,
    GaussianNoiseModel,
    ConstantRewardBandit,
)
from single_peaked_bandits.solvers import (
    GreedySolver,
    IncreasingSolver,
    OptimisticSolver,
    Exp3Solver,
    OneStepOptimisticSolver,
    SlidingWindowUCB,
    DiscountedUCB,
    Rexp3,
    UCB,
)
from single_peaked_bandits.constants import FICO_EXP_FOLDER


def exp1_f1(t):
    return 1 - t ** (-0.5) if t > 0 else 0


def exp1_f2(alpha, t):
    return 0.5 - 0.5 * t ** (-alpha) if t > 0 else 0


def exp2_f1(t):
    return min(1, t / 1000)


def exp2_f2(alpha, t):
    return min(0.5, 0.5 * (t / 1000) ** alpha)


def exp3_f1(t):
    return 1 - t ** (-0.1) if t > 0 else 0


def exp3_f2(alpha, t):
    return 0.5 - 0.5 * t ** (-alpha) if t > 0 else 0


def exp4_f(k1, k2, c1, c2, l, a, t):
    return (
        a * np.exp(-k1 * (t - l)) + (c2 - c1) / (np.exp(-(k1 + k2) * (t - l)) + 1) + c1
    )


Tmax = 10000

EXPERIMENT_INC_1_BANDITS = [
    Bandit(
        f"inc_1_alpha_{alpha}",
        [exp1_f1, partial(exp1_f2, alpha)],
        Tmax,
        increasing=True,
    )
    for alpha in (0.1, 0.5, 1, 5)
]
EXPERIMENT_INC_2_BANDITS = [
    Bandit(
        f"inc_2_alpha_{alpha}",
        [exp2_f1, partial(exp2_f2, alpha)],
        Tmax,
        increasing=True,
    )
    for alpha in (0.03, 0.1, 0.4, 1)
]
EXPERIMENT_INC_3_BANDITS = [
    Bandit(
        f"inc_3_alpha_{alpha}",
        [exp3_f1, partial(exp3_f2, alpha)],
        Tmax,
        increasing=True,
    )
    for alpha in (0.1, 0.5, 1, 5)
]

EXPERIMENT_INC_1_GAUSSIAN_NOISE_BANDITS = [
    Bandit(
        f"inc_1_alpha_{alpha}_std_{std}",
        [exp1_f1, partial(exp1_f2, alpha)],
        Tmax,
        increasing=True,
        noise_model=GaussianNoiseModel(std),
    )
    for alpha in (0.1, 0.5, 1, 5)
    for std in (0.01, 0.05, 0.1)
]
EXPERIMENT_INC_2_GAUSSIAN_NOISE_BANDITS = [
    Bandit(
        f"inc_2_alpha_{alpha}_std_{std}",
        [exp2_f1, partial(exp2_f2, alpha)],
        Tmax,
        increasing=True,
        noise_model=GaussianNoiseModel(std),
    )
    for alpha in (0.03, 0.1, 0.4, 1)
    for std in (0.01, 0.05, 0.1)
]
EXPERIMENT_INC_3_GAUSSIAN_NOISE_BANDITS = [
    Bandit(
        f"inc_3_alpha_{alpha}_std_{std}",
        [exp3_f1, partial(exp3_f2, alpha)],
        Tmax,
        increasing=True,
        noise_model=GaussianNoiseModel(std),
    )
    for alpha in (0.1, 0.5, 1, 5)
    for std in (0.01, 0.05, 0.1)
]


EXPERIMENT_INC_NOISE_BANDITS = [
    Bandit(
        "inc_2_alpha_0.1_sigma_0",
        [exp2_f1, partial(exp2_f2, 0.1)],
        Tmax,
        increasing=True,
    )
] + [
    Bandit(
        f"inc_2_alpha_0.1_sigma_{sigma}",
        [exp2_f1, partial(exp2_f2, 0.1)],
        Tmax,
        increasing=True,
        noise_model=SymmetricBetaNoiseModel(0.1, sigma),
    )
    for sigma in (0.02, 0.04, 0.06, 0.08)
]


Tmax = 20000
EXPERIMENT_INC_DEC_BANDITS = [
    Bandit(
        "inc_dec_1",
        [
            partial(exp4_f, 0.01, 0.001, 1, 0.05, 600, -0.0015),
            partial(exp4_f, 0.009, 0.0009, 0.8, 0.1, 500, -0.005),
        ],
        Tmax,
    ),
    Bandit(
        "inc_dec_2",
        [
            partial(exp4_f, 0.003, 0.001, 1, 0.05, 600, -0.0015),
            partial(exp4_f, 0.011, 0.001, 0.8, 0.2, 400, -0.008),
        ],
        Tmax,
    ),
    Bandit(
        "inc_dec_3",
        [
            partial(exp4_f, 0.01, 0.001, 1, 0.5, 600, -0.0015),
            partial(exp4_f, 0.009, 0.0009, 0.8, 0.6, 500, -0.005),
        ],
        Tmax,
    ),
]

EXPERIMENT_INC_DEC_NOISE_BANDITS = (
    [
        Bandit(
            "inc_dec_1_sigma_0",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.1, 500, -0.005),
            ],
            Tmax,
        )
    ]
    + [
        Bandit(
            f"inc_dec_1_sigma_{sigma}",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.1, 500, -0.005),
            ],
            Tmax,
            increasing=False,
            noise_model=SymmetricBetaNoiseModel(0.1, sigma),
        )
        for sigma in (0.02, 0.04, 0.06, 0.08)
    ]
    + [
        Bandit(
            "inc_dec_2_sigma_0",
            [
                partial(exp4_f, 0.003, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.011, 0.001, 0.8, 0.2, 400, -0.008),
            ],
            Tmax,
        )
    ]
    + [
        Bandit(
            f"inc_dec_2_sigma_{sigma}",
            [
                partial(exp4_f, 0.003, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.011, 0.001, 0.8, 0.2, 400, -0.008),
            ],
            Tmax,
            increasing=False,
            noise_model=SymmetricBetaNoiseModel(0.1, sigma),
        )
        for sigma in (0.02, 0.04, 0.06, 0.08)
    ]
    + [
        Bandit(
            "inc_dec_3_sigma_0",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.5, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.6, 500, -0.005),
            ],
            Tmax,
        )
    ]
    + [
        Bandit(
            f"inc_dec_3_sigma_{sigma}",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.5, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.6, 500, -0.005),
            ],
            Tmax,
            increasing=False,
            noise_model=SymmetricBetaNoiseModel(0.1, sigma),
        )
        for sigma in (0.02, 0.04, 0.06, 0.08)
    ]
)


EXPERIMENT_INC_DEC_GAUSSIAN_NOISE_BANDITS = (
    [
        Bandit(
            f"inc_dec_1_std_{std}",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.1, 500, -0.005),
            ],
            Tmax,
            increasing=False,
            noise_model=GaussianNoiseModel(std),
        )
        for std in (0.01, 0.05, 0.1)
    ]
    + [
        Bandit(
            f"inc_dec_2_std_{std}",
            [
                partial(exp4_f, 0.003, 0.001, 1, 0.05, 600, -0.0015),
                partial(exp4_f, 0.011, 0.001, 0.8, 0.2, 400, -0.008),
            ],
            Tmax,
            increasing=False,
            noise_model=GaussianNoiseModel(std),
        )
        for std in (0.01, 0.05, 0.1)
    ]
    + [
        Bandit(
            f"inc_dec_3_std_{std}",
            [
                partial(exp4_f, 0.01, 0.001, 1, 0.5, 600, -0.0015),
                partial(exp4_f, 0.009, 0.0009, 0.8, 0.6, 500, -0.005),
            ],
            Tmax,
            increasing=False,
            noise_model=GaussianNoiseModel(std),
        )
        for std in (0.01, 0.05, 0.1)
    ]
)


Tmax = 5000

EXPERIMENT_CONST_BANDITS = [
    ConstantRewardBandit(
        f"const_1",
        n_arms=10,
        Tmax=Tmax,
        noise_model=None,
        seed=9,
    )
]

EXPERIMENT_CONST_GAUSSIAN_NOISE_BANDITS = [
    ConstantRewardBandit(
        f"const_1_std_{std}",
        n_arms=10,
        Tmax=Tmax,
        noise_model=GaussianNoiseModel(std),
        seed=9,
    )
    for std in (0.01, 0.05, 0.1)
]


fico_groups = ["Asian", "Black", "Hispanic", "White"]
fico_group_pairs = itertools.combinations(fico_groups, 2)
Tmax = 2000
EXPERIMENT_FICO_BANDITS_TWO_ARMS = sum(
    [
        [
            DataBandit(
                f"fico_{group_A}_{group_B}_utility",
                [
                    os.path.join(
                        FICO_EXP_FOLDER,
                        f"fico_reward_group_{label}_utility.npy",
                    )
                    for label in (group_A, group_B)
                ],
                Tmax,
                SymmetricBetaNoiseModel(0.1, 0.02),
            ),
            DataBandit(
                f"fico_{group_A}_{group_B}_score_change",
                [
                    os.path.join(
                        FICO_EXP_FOLDER,
                        f"fico_reward_group_{label}_score_change.npy",
                    )
                    for label in (group_A, group_B)
                ],
                Tmax,
                SymmetricBetaNoiseModel(0.1, 0.02),
            ),
        ]
        for group_A, group_B in fico_group_pairs
    ],
    [],
)
EXPERIMENT_FICO_BANDITS_ALL_ARMS = [
    DataBandit(
        "fico_all_utility",
        [
            os.path.join(FICO_EXP_FOLDER, f"fico_reward_group_{label}_utility.npy")
            for label in fico_groups
        ],
        Tmax,
        SymmetricBetaNoiseModel(0.1, 0.02),
        inject_noise=False,
    ),
    DataBandit(
        "fico_all_score_change",
        [
            os.path.join(FICO_EXP_FOLDER, f"fico_reward_group_{label}_score_change.npy")
            for label in fico_groups
        ],
        Tmax,
        SymmetricBetaNoiseModel(0.1, 0.02),
        inject_noise=False,
    ),
]
EXPERIMENT_FICO_BANDITS_ALL_ARMS_GAUSSIAN_NOISE = [
    DataBandit(
        f"fico_all_utility_gaussian_noise_{std}",
        [
            os.path.join(FICO_EXP_FOLDER, f"fico_reward_group_{label}_utility.npy")
            for label in fico_groups
        ],
        Tmax,
        GaussianNoiseModel(std, bound=max(0.1, 2 * std)),
        inject_noise=True,
    )
    for std in (0.01, 0.05, 0.1)
] + [
    DataBandit(
        f"fico_all_score_change_gaussian_noise_{std}",
        [
            os.path.join(FICO_EXP_FOLDER, f"fico_reward_group_{label}_score_change.npy")
            for label in fico_groups
        ],
        Tmax,
        GaussianNoiseModel(std, bound=max(0.1, 2 * std)),
        inject_noise=True,
    )
    for std in (0.01, 0.05, 0.1)
]

Tmax = 3000
EXPERIMENT_RECOMMENDER_2_BANDITS = [
    RecommenderSystemBandit.get_random(f"recommender_2_{seed}", Tmax, 2, seed=seed)
    for seed in [1, 2, 3]
]
Tmax = 3000
EXPERIMENT_RECOMMENDER_4_BANDITS = [
    RecommenderSystemBandit.get_random(f"recommender_4_{seed}", Tmax, 4, seed=seed)
    for seed in [1, 2, 3]
]
EXPERIMENT_RECOMMENDER_4_GAUSSIAN_NOISE_BANDITS = [
    RecommenderSystemBandit.get_random(
        f"recommender_4_{seed}_gaussian_noise_{std}",
        Tmax,
        4,
        noise_model=GaussianNoiseModel(std),
        seed=seed,
    )
    for seed in [1, 2, 3]
    for std in [0.01, 0.05, 0.1]
]


EXPERIMENTS = {
    "inc_1": (
        EXPERIMENT_INC_1_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_2": (
        EXPERIMENT_INC_2_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_3": (
        EXPERIMENT_INC_3_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_1_gaussian_noise": (
        EXPERIMENT_INC_1_GAUSSIAN_NOISE_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_2_gaussian_noise": (
        EXPERIMENT_INC_2_GAUSSIAN_NOISE_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_3_gaussian_noise": (
        EXPERIMENT_INC_3_GAUSSIAN_NOISE_BANDITS,
        (
            IncreasingSolver,
            OptimisticSolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_dec_1": (
        EXPERIMENT_INC_DEC_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_noise": (
        EXPERIMENT_INC_NOISE_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_dec_noise": (
        EXPERIMENT_INC_DEC_NOISE_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "inc_dec_gaussian_noise": (
        EXPERIMENT_INC_DEC_GAUSSIAN_NOISE_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "const_1": (
        EXPERIMENT_CONST_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            UCB
        ),
    ),
    "const_1_gaussian_noise": (
        EXPERIMENT_CONST_GAUSSIAN_NOISE_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            UCB
        ),
    ),
    "fico_two": (
        EXPERIMENT_FICO_BANDITS_TWO_ARMS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "fico_all": (
        EXPERIMENT_FICO_BANDITS_ALL_ARMS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "fico_all_gaussian_noise": (
        EXPERIMENT_FICO_BANDITS_ALL_ARMS_GAUSSIAN_NOISE,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "recommender_2": (
        EXPERIMENT_RECOMMENDER_2_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "recommender_4": (
        EXPERIMENT_RECOMMENDER_4_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
    "recommender_4_gaussian_noise": (
        EXPERIMENT_RECOMMENDER_4_GAUSSIAN_NOISE_BANDITS,
        (
            OptimisticSolver,
            OneStepOptimisticSolver,
            GreedySolver,
            Exp3Solver,
            SlidingWindowUCB,
            DiscountedUCB,
            Rexp3,
        ),
    ),
}

ALL_BANDITS = dict()
for experiment in EXPERIMENTS.values():
    bandits = experiment[0]
    for bandit in bandits:
        if bandit.name not in ALL_BANDITS:
            ALL_BANDITS[bandit.name] = bandit

if __name__ == "__main__":
    from make_plots import plot_rewards
    import matplotlib.pyplot as plt

    bandits = EXPERIMENT_FICO_BANDITS_ALL_ARMS_GAUSSIAN_NOISE

    plt.figure()
    for bandit in bandits:
        plot_rewards(plt.gca(), bandit)
        plt.show()
