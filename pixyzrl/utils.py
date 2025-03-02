from typing import Any

import pixyz
from IPython.display import Math


def is_env_notebook() -> bool:
    """Determine wheather is the environment Jupyter Notebook

    Returns:
        bool: True if the environment is Jupyter Notebook, False otherwise.

    Examples:
        >>> is_env_notebook()
        False
    """
    if "get_ipython" not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__  # type: ignore # noqa: F821, PGH003
    # Return the negated condition directly
    return env_name != "TerminalInteractiveShell"


def print_latex(obj: Any) -> Math | str | None:
    """Print formulas in latex format.

    Args:
        obj (Any): Object to be printed in latex format.

    Returns:
        Math | str | None: Math object if the environment is Jupyter Notebook, string if not, None otherwise.

    Examples:
        >>> from pixyz.distributions import Normal
        >>> from pixyz.losses import KullbackLeibler
        >>> p = Normal(loc=0., scale=1., var=["x"], cond_var=["y"], features_shape=[1], name="p")
        >>> q = Normal(loc=0., scale=1., var=["x"], cond_var=["y"], features_shape=[1], name="q")
        >>> print_latex(KullbackLeibler(p, q))
        D_{KL} \\left[p(x|y)||q(x|y) \\right]
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
