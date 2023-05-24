import numpy as np
from numpy import ndarray
from typing import Tuple, Any

from constants import e


class SET:
    def __init__(self, C1: float, C2: float, V: ndarray, Vg: ndarray = 0, Cg: float = 0, Q0: float = 0.0, T: float = 0.0, alpha: float = 0.0):
        """
        Initialization of the SET parameter
        :param C1: Capacitance of first junction (F)
        :param C2: Capacitance of second junction (F)
        :param V: External potential (V),
        :param Vg: Gate control potential (V), by default 0 meaning no Cg,Vg will be taken into account
        :param Cg: Control gate capacitance (F), by default 0, meaning no Cg,Vg will be taken into account
        :param Q0: Background charge, 0 by default
        :param T: Temperature (K) of the system, 0 temperature by default
        :param alpha: Lever parameter, factor depending on the circuit, 0 by default
        """
        # Circuit factor
        self.alpha = alpha

        # Potential
        self.V = V

        # Temperature
        self.T = T

        # Gate voltage
        self.Vg = Vg
        # Gate capacitance
        self.Cg = Cg
        # Junctions capacitance
        self.C1 = C1 + self.alpha * self.Cg
        self.C2 = C2 + (1 - self.alpha) * self.Cg
        # Total capacitance
        self.C_total = self.C1 + self.C2 + self.Cg

        # Background charge
        self.Q0 = Q0
        self.Qext = self.Q0 + self.Cg * (self.Vg - self.alpha * self.V)

    def free_energy(self, N1: float, N2: float) -> ndarray:
        """
        Calculate the free energy for N1 electrons at junction 1 and N2 electron at junction 2
        :param N1: Number of electrons at junction 1
        :param N2: Number of electrons at junction 2
        :return: Free energy, as a numpy array
        """
        N = N1 - N2
        V, Vg = np.meshgrid(self.V, self.Vg)
        electrostatic_energy = (self.C1 + self.Cg) * self.C2 * (V ** 2) + (N * e - (self.Q0 + self.Cg * (Vg - self.alpha * V))) / (2 * self.C_total)
        potential_work = - (N1 * self.C2 + N2 * (self.C1 + self.Cg)) * e * V / self.C_total

        F = electrostatic_energy + potential_work
        return F

    def energy_diff(self, N1: float, N2: float) -> Tuple:
        """
        Calculate the energy difference when the number of electrons fluctuates between N1±1 and N2±1
        :param N1: Number of electron at junction 1
        :param N2: Number of electron at junction 2
        :return: Energy difference for each of the 4 cases (N1 + 1 | N1 - 1 | N2 + 1 | N2 - 1)
        """
        N = N1 - N2
        V, Vg = np.meshgrid(self.V, self.Vg)

        delta_free_energy_plus_N1 = e / self.C_total * (
                    e / 2 + (N * e - self.Cg * Vg - self.Qext + self.C2 * V))
        delta_free_energy_minus_N1 = e / self.C_total * (
                    e / 2 - (N * e - self.Cg * Vg - self.Qext + self.C2 * V))
        delta_free_energy_plus_N2 = e / self.C_total * (
                    e / 2 + (-N * e + self.Cg * Vg + self.Qext + (self.C1 + self.Cg) * V))
        delta_free_energy_minus_N2 = e / self.C_total * (
                    e / 2 - (-N * e + self.Cg * Vg + self.Qext + (self.C1 + self.Cg) * V))

        return delta_free_energy_plus_N1, delta_free_energy_minus_N1, delta_free_energy_plus_N2, delta_free_energy_minus_N2
