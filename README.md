# Addressing the Long-term Impact of ML-based Decisions via Policy Regret

This repository contains code to reproduce the experiments of the paper "Addressing the Long-term Impact of ML-based Decisions via Policy Regret'. This code is provided as is, and will not be maintained. Here we describe how to reproduce the experimental results reported in the paper.

### Citation

David Lindner, Hoda Heidari, Andreas Krause. **Addressing the Long-term Impact of ML-based Decisions via Policy Regret**. In _International Joint Conference on Artificial Intelligence (IJCAI)_, 2021.

```
@inproceedings{lindner2021addressing
    title={Addressing the Long-term Impact of ML-based Decisions via Policy Regret},
    author={Lindner, David and Heidari, Hoda and Krause, Andreas},
    booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2021},
}
```

## Setup

The code has been tested with the Anaconda distribution of Python version 3.6.8.

To set up such an environment install [Anaconda](https://www.anaconda.com/), and run the following commands:
```
conda create -n single-peaked-bandits python=3.6.8
conda activate single-peaked-bandits
pip install -e .
conda install -c gurobi gurobi
```

The last command is optional for using [Gurobi](https://www.gurobi.com/) instead of [Cbc](https://github.com/coin-or/Cbc) to solve linear programs. Gurobi will be faster, but requires a license.

## Reproducing the experiments

First run the following command to preprocess the FICO dataset:
```
python src/single_peaked_bandits/fico_experiment/preprocessing.py
```

Then, use the following commands to run the experiments reported in the paper:
```
# Syntethic reward functions
## Increasing reward function in Appendix D.1
python src/run_experiments.py --labels "inc_1,inc_2,inc_3,inc_1_gaussian_noise"
## Single-peaked reward functions in Figure 1 and Appendix D.2
python src/run_experiments.py --labels "inc_dec_gaussian_noise"
## Gaussian multi-armed bandits in Appendix D.3
python src/run_experiments.py --labels "const_1_gaussian_noise"

# FICO dataset (Figure 3 and Appendix D.4)
python src/run_experiments.py --labels "fico_all_gaussian_noise"

# Recommender system simulation (Figure 3 and Appendix D.5)
python src/run_experiments.py --labels "recommender_4_gaussian_noise"
```

Running each command automatically creates a series of plots in `results/plots`, and the full results are written to `results/pickle`. Plots can also be created manually using the `src/make_plots.py` script.

To reduce computational cost you can reduce the number of experiments with the command line parameters `--n_discretization` and `--n_seeds`. You can parallelize the experiments using `--n_jobs`.

To compute regret, the optimal policies have to be compute, which can be done by running the command for a given experiment above with the `--optimal` flag. This precomputes the optimal policies which can then be used to compute regret, which can be useful because computing the optimal policy is expensive in some environments.

Alternatively, the regret can be directly computed by passing `--compute_regret` to `src/run_experiments.py`, which can be useful for randomly sampled reward functions.
