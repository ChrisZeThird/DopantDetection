from constants import e, hbar
from math import pi
from typing import Any

from utils.fermi import fermi_autocorrelation


def tunneling_conductance(rho1: float = 1, rho2: float = 1, t: float = 0.5) -> float:
    """
    Calculate the tunneling conductance of a Single Electron Transistor system
    Based on "Single electron transport and single dopant detection in silicon transistors", Mathieu Pierre, 2010
    :param rho1: Density of states in the electrode
    :param rho2: Density of states in the island
    :param t:  Tunnel coefficient
    :return: Conductance G
    """
    return (e**2 / hbar) * 4 * (pi ** 2) * rho1 * rho2 * abs(t)**2


def tunneling_rate(delta_electrostatic: Any, V: Any, T: float = 0) -> Any:
    """
    Calculate the tunneling rate of a SET system. The default case set temperature at 0 Kelvin.
    Based on "Single electron transport and single dopant detection in silicon transistors", Mathieu Pierre, 2010
    :param delta_electrostatic:
    :param V: Voltage of the electrode
    :param T: Temperature
    :return: Tunneling rate Gamma
    """
    tunnel_conductance = tunneling_conductance()
    if T != 0:
        return (tunnel_conductance / (e ** 2)) * fermi_autocorrelation(delta_electrostatic + e * V)
