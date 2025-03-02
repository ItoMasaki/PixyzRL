import re
from typing import Any

import pixyz
from IPython.display import Math


def is_env_notebook() -> bool:
    """Determine wheather is the environment Jupyter Notebook"""
    if "get_ipython" not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__
    # Return the negated condition directly
    return env_name != "TerminalInteractiveShell"


def print_latex(obj: Any) -> Math | str | None:
    """Print formulas in latex format.

    Parameters
    ----------
    obj : pixyz.distributions.distributions.Distribution, pixyz.losses.losses.Loss or pixyz.models.model.Model.

    """

    if isinstance(obj, pixyz.distributions.distributions.Distribution | pixyz.distributions.distributions.DistGraph):
        latex_text = obj.prob_joint_factorized_and_text
    elif isinstance(obj, pixyz.losses.losses.Loss):
        latex_text = obj.loss_text
    elif isinstance(obj, pixyz.models.model.Model):
        latex_text = obj.loss_cls.loss_text

    if is_env_notebook():
        return Math(latex_text)

    print(latex_text)  # noqa: T201
    return None
