from math import pi
from numpy import ndarray
import numpy as np
from typing import Any

import warnings

from constants import e, kB
from utils.fermi import fermi

# TODO there are a HUGE number of chemical potential states but transitions between most of them aren’t allowed because
#  BOTH charge states (the N and the M) have to be consistent. For example the transition between (4,5) -> (2,6) are not
#  going to be allowed. Even though you’re just adding one electron to the 2nd dot you’re removing 2 from the first?


class DoubleDot:
    def __init__(self, Vg1: ndarray = 0, Cg1: float = 0, Vg2: ndarray = 0, Cg2: float = 0, CL: float = 0, RL: float = 0, CR: float = 0, RR: float = 0, Cm: float = 0, Rm: float = 0, T: float = 0.01):
        """
        Initialization of the SET parameter. Definition is confusing here as C1 and C1 are source and rain capacitance,
        and not of dot 1 and dot 2.
        Grand canonical ensemble approach is used in order to obtain the distribution of electrons in the double dots. The
        code is based on the work of Pierre-Antoine Mouny at the 3iT, at the University of Sherbrooke.
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

        # Electrostatic coupling (charging energy)
        self.EC1 = (e ** 2) * (self.C2 / (self.C1 * self.C2 - (self.Cm ** 2)))
        self.EC2 = (e ** 2) * (self.C1 / (self.C1 * self.C2 - (self.Cm ** 2)))
        self.ECm = (e ** 2) * (self.Cm / (self.C1 * self.C2 - (self.Cm ** 2)))

    def electrostatic_energy(self, N1: int, N2: int):
        """
        Double dot electrostatic energy. Based on equation (1) in 'Electron transport through double quantum dots', by
        W. G van der Wiel, 2003
        :param N1: Number of electron on dot 1
        :param N2: Number of electron on dot 2
        :return:
        """
        f = -1 / e * (self.Cg1 * self.Vg1 * (N1 * self.EC1 + N2 * self.ECm) + self.Cg2 * self.Vg2 * (N1 * self.ECm + N2 * self.EC2))\
            + 1 / (e ** 2) * (0.5 * (self.Cg1 ** 2) * (self.Vg1 ** 2) * self.EC1 + 0.5 * (self.Cg2 ** 2) * (self.Vg2 ** 2) * self.EC2
                              + self.Cg1 * self.Vg1 * self.Cg2 * self.Vg2 * self.ECm)

        U = 0.5 * (N1 ** 2) * self.EC1 + 0.5 * (N2 ** 2) * self.EC2 + 0.5 * (N1 * N2) * self.ECm + f
        return U

    def electron_statistic(self, N1, N2):
        """
        Calculate the statistical distribution of electrons in the double quantum dots, considered as a Grand Canonical
        Ensemble.
        :param N1: Number of electrons in dot 1
        :param N2: Number of electrons in dot 2
        :return: The weighted average of electrons number in the DQD
        """

        partition_func = 0
        avg_nbr = 0
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for i in range(0, N2 + 1):
                for j in range(0, N1 + 1):
                    elec_energy = self.electrostatic_energy(j, i)
                    partition_func = partition_func + np.exp(-elec_energy / (kB * self.T))
                    avg_nbr = avg_nbr + (j - i) * np.exp(-elec_energy / (kB * self.T))

            return avg_nbr / partition_func
