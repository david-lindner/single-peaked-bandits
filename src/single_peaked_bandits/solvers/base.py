class BaseSolver:
    stochastic = False

    def __init__(self, name):
        self.name = name

    def solve(self, bandit, T):
        raise NotImplementedError()
