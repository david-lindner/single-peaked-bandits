from .greedy import GreedySolver
from .increasing import IncreasingSolver
from .optimistic import OptimisticSolver
from .exp3 import Exp3Solver
from .one_step_optimistic import OneStepOptimisticSolver
from .sliding_window_ucb import SlidingWindowUCB
from .discounted_ucb import DiscountedUCB
from .r_exp3 import Rexp3
from .optimal import OptimalSolver
from .ucb import UCB

SOLVERS = {
    "greedy": GreedySolver,
    "increasing": IncreasingSolver,
    "optimistic": OptimisticSolver,
    "exp3": Exp3Solver,
    "one-ste-optimistic": OneStepOptimisticSolver,
    "sucb": SlidingWindowUCB,
    "ducb": DiscountedUCB,
    "Rexp3": Rexp3,
    "optimal": OptimalSolver,
    "ucb": UCB,
}
