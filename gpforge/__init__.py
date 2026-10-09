"""gpforge — world-class deterministic Gaussian Process toolkit.

Flagship :meth:`gpforge.gp.methods.ExactGP` (RKHS-kernel regression with
Cholesky inference and automatic hyperparameter optimization) and
:meth:`gpforge.gp.methods.BayesOpt` (Expected-Improvement / Upper-Confidence-Bound
acquisition on top of the GP posterior).

Author: 晨星 (Morning Star). License: MIT.
"""

__version__ = "0.1.0"
__author__ = "晨星"

from gpforge.core.config import Config
from gpforge.core.seed import set_all

__all__ = ["Config", "__author__", "__version__", "set_all"]
