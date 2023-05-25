from numpy import exp
from typing import Any
from scipy.special import expit

from constants import kB


def fermi(x: Any) -> Any:
    """
    Fermi dirac distribution function
    :param x: Any type
    :return: Fermi dirac statistic
    """
    return expit(-x)


def fermi_autocorrelation(E: Any, T: float = 0.0) -> Any:
    """
    Fermi autocorrelation function
    :param E: Energy
    :param T: Temperature, by default set at 0 Kelvin
    :return:
    """
    if T != 0:
        return E / (exp(E / (kB * T) - 1))
    else:
        return E
