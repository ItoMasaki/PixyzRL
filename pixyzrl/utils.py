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
    env_name = get_ipython().__class__.__name__
    # Return the negated condition directly
    return env_name != "TerminalInteractiveShell"


def print_latex(obj: Any) -> Math | str | None:
    """Print formulas in latex format.

    Args:
        obj (Any): Object to be printed in latex format.

    Returns:
        Math | str | None: Math object if the environment is Jupyter Notebook, string if not, None otherwise.

    Examples:
        >>> print_latex(pixyz.losses.KullbackLeibler())
        \begin{equation}KL\left[p(x)||q(x)\right] = \mathbb{E}_{p(x)}\left[\log\frac{p(x)}{q(x)}\right]\end{equation}
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
