from math import pi
from numpy import ndarray
import numpy as np
from typing import Any

from constants import e, kB
from utils.fermi import fermi

# TODO there are a HUGE number of chemical potential states but transitions between most of them aren’t allowed because
#  BOTH charge states (the N and the M) have to be consistent. For example the transition between (4,5) -> (2,6) are not
#  going to be allowed. Even though you’re just adding one electron to the 2nd dot you’re removing 2 from the first?


class SET:
    def __init__(self, Vg1: ndarray = 0, Cg1: float = 0, Vg2: ndarray = 0, Cg2: float = 0, CL: float = 0, RL: float = 0, CR: float = 0, RR: float = 0, Cm: float = 0, Rm: float = 0, T: float = 0.01):
        """
        Initialization of the SET parameter. Definition is confusing here as C1 and C1 are source and rain capacitance,
        and not of dot 1 and dot 2.
        We consider the linear transport regime, i.e., bias voltage V=0 in the first place.
        :param Vg1: Gate control potential on dot 1 (V), by default 0 meaning no Cg1,Vg1 will be taken into account
        :param Cg1: Control gate capacitance on dot 1 (F) ------------------------//-------------------------------
        :param Vg2: Gate control potential on dot 2 (V), by default 0 meaning no Cg2,Vg2 will be taken into account
        :param Cg2: Control gate capacitance on dot 2 (F) ------------------------//-------------------------------
        :param CL: Capacitance of left junction (source)
        :param RL: Tunnel resistor of left junction (source)
        :param CR: Capacitance of right junction (drain)
        :param RR: Tunnel resistor of right junction (drain)
        :param Cm: Coupling capacitance between the dots
        :param Rm: Tunnel resistor between the dots
        :param T: Temperature (K) of the system, 0.01 temperature by default to avoid division by zero
        """
        # Junctions parameters
        self.CL = CL  # left junction capacitance (source)
        self.RL = RL  # -----//----- tunnel resistor (source)

        self.CR = CR  # left junction capacitance (drain)
        self.RR = RR  # -----//----- tunnel resistor (drain)

        # Gate parameters on dot 1 and dot 2
        self.Vg1 = Vg1
        self.Cg1 = Cg1

        self.Vg2 = Vg2
        self.Cg2 = Cg2

        # Dot coupling parameters
        self.Cm = Cm
        self.Rm = Rm

        # Dot 1 capacitance
        self.C1 = self.CL + self.Cg1 + self.Cm

        # Dot 2 capacitance
        self.C2 = self.CR + self.Cg2 + self.Cm

        # Temperature
        self.T = T

        # Electrostatic coupling
        self.EC1 = self.charging_energy_dot(self.C1)
        self.EC2 = self.charging_energy_dot(self.C2)
        self.ECm = self.charging_energy_dot(self.Cm)

    def charging_energy_dot(self, capacitance):
        """
        Calculates the charging energy of the individual dot considered with capacitance `capacitance`
        :param capacitance:
        :return:
        """
        if capacitance != self.Cm:
            return (e ** 2) / capacitance * (1 / (1 - (self.Cm ** 2) / (self.C1 * self.C2)))
        else:
            return (e ** 2) / capacitance * (1 / ((self.C1 * self.C2) / (self.Cm ** 2) / - 1))

    def electrostatic_energy(self, N1, N2):
        """
        Double dot electrostatic energy. Based on equation (1) in 'Electron transport through double quantum dots', by
        W. G van der Wiel, 2003
        :param N1: Number of electron on dot 1
        :param N2: Number of electron on dot 2
        :return:
        """
        f = 1 / (- e) * (self.Cg1 * self.Vg1 * (N1 * self.EC1 + N2 * self.ECm) + self.Cg2 * self.Vg2 * (N1 * self.ECm + N2 * self.EC2)) + 1 / (e ** 2) * (
                0.5 * (self.Cg1 ** 2) * (self.Vg1 ** 2) * self.EC1 + 0.5 * (self.Cg2 ** 2) * (self.Vg2 ** 2) * self.EC2 + self.Cg1 * self.Vg1 * self.Cg2 * self.Vg2 * self.ECm)

        U = 0.5 * (N1 ** 2) * self.EC1 + 0.5 * (N2 ** 2) * self.EC2 + 0.5 * (N1 * N2) * self.ECm + f
        return U

    def rate(self, energy_diff, mu, distribution: callable):
        """
        Calculate rate between two states
        :param energy_diff: Energy difference between states
        :param mu: Chemical potential
        :param distribution: The distribution function, either fermi or 1 - fermi
        :return: Rate between the two states
        """
        return self.RL * distribution((energy_diff - mu) * self.beta_ev) + self.RR * distribution(
            (energy_diff - mu) * self.beta_ev)

    def current(self, states_number_dot1: int = 5, states_number_dot2: int = 5) -> ndarray:
        """
        Calculate the current through the double quantum dot for given number of states for each dot
        :param states_number_dot1: Number of states to consider for the first dot (N1)
        :param states_number_dot2: Number of states to consider for the second dot (N2)
        :return: Current matrix, as a 2D numpy array
        """

        if states_number_dot1 < 1 or states_number_dot2 < 1:
            raise ValueError('Number of states cannot be smaller than 1.')

        Vg1_2d, Vg2_2d = np.meshgrid(self.Vg1, self.Vg2)

        # Initialise current matrix
        current_matrix = np.zeros((len(self.Vg1), len(self.Vg2)))

        # Matrices from equation C = AP
        A = np.zeros((states_number_dot1 + 1, states_number_dot2 + 1))  # matrix A from the master equation of
        A[states_number_dot1, :] = 1  # normalize all probabilities to 1 on last row



