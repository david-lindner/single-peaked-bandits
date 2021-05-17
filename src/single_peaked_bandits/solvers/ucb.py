import numpy as np

from single_peaked_bandits.helpers import cumulative_reward
from single_peaked_bandits.solvers.discounted_ucb import DiscountedUCB


class UCB(DiscountedUCB):
    def __init__(self, xi=0.6):
        super().__init__(xi=xi, gamma=1)
        self.name = "ucb"
