import numpy as np
from numpy import ndarray
from typing import Tuple, Any

from constants import e, kB
from utils.fermi import fermi


class SET:
    def __init__(self, C1: float, C2: float, Vd: ndarray, Vs: float = 0.0, Vg: ndarray = 0, Cg: float = 0, Q0: float = 0.0, RL: float = 1.6e6, RR: float = 1.6e6, T: float = 0.01, alpha: float = 0.0):
        """
        Initialization of the SET parameter
        :param C1: Capacitance of first junction (F)
        :param C2: Capacitance of second junction (F)
        :param Vd: Drain potential (V)
        :param Vs: Source potential (V)
        :param Vg: Gate control potential (V), by default 0 meaning no Cg,Vg will be taken into account
        :param Cg: Control gate capacitance (F), by default 0, meaning no Cg,Vg will be taken into account
        :param Q0: Background charge, 0 by default
        :param RL: Tunnel rate of left barrier, 1.6e6 by default
        :param RR: Tunnel rate of right barrier, 1.6e6 by default
        :param T: Temperature (K) of the system, 0.01 temperature by default to avoid division by zero
        :param alpha: Lever parameter, factor depending on the circuit, 0 by default
        """
        # Circuit factor
        self.alpha = alpha

        # Temperature
        self.T = T
        self.beta = 1 / (kB * self.T)

        # Drain voltage
        self.Vd = Vd
        # Source Voltage
        self.Vs = Vs
        # Bias potential
        self.V = self.Vs - self.Vd
        # Gate voltage
        self.Vg = Vg

        # Gate capacitance
        self.Cg = Cg
        # Junctions capacitance
        self.C1 = C1 + self.alpha * self.Cg
        self.C2 = C2 + (1 - self.alpha) * self.Cg
        # Total capacitance
        self.C_total = self.C1 + self.C2 + self.Cg

        # Tunnel rates or left (L) and right (R) barriers
        self.RL = RL
        self.RR = RR

        # Background charge
        self.Q0 = Q0
        self.Qext = self.Q0 + self.Cg * (self.Vg - self.alpha * self.V)

        # Charging energy
        self.Ec = e ** 2 / self.C_total
        # Chemical potential
        self.muL = -e * self.Vs  # left barrier (source)

        #

    def state_energy(self, N):
        """
        Calculate energy of state N
        :param N: Index of the state
        :return:
        """
        return (-e * (N - self.Q0 - 0.5) + self.C1 * self.Vs + self.C2 * self.Vd + self.Cg * self.Vg) ** 2 / (2 * e * self.C_total)

    def rate(self, energy_diff, distribution: callable):
        """
        Calculate rate between two states
        :param energy_diff: Energy difference between states
        :param distribution: The distribution function, either fermi or 1 - fermi
        :return:
        """
        return self.RL * distribution((energy_diff - self.muL) * self.beta) + self.RR * distribution((energy_diff - self.muL) * self.beta)

    def current(self, states_number: int = 5) -> ndarray:
        """
        Calculate the current through the device for N states available
        :param states_number: Number of states to consider (N=5 by default)
        :return: Free energy, as a list of numpy array from E0 to E5 (or E_N for N set by states_number)
        """
        if states_number < 1:
            raise ValueError('Number of states cannot be smaller than 1.')

        # Initialise current matrix
        current_matrix = np.full((len(self.Vg), len(self.Vd)), e)

        # Calculate for N=0 and N=1 the energies and the rates
        state_energy_0 = self.state_energy(0)
        state_energy_1 = self.state_energy(1)
        energy_diff_10 = state_energy_1 - state_energy_0
        rate_10 = self.rate(energy_diff_10, fermi)
        rate_01 = self.rate(energy_diff_10, lambda E: 1 - fermi(E))

        energy_list = [state_energy_0, state_energy_1]
        energy_diff_list = [energy_diff_10]
        rates_list = [(rate_10, rate_01)]  # initialise list to store tuples of rates, e.g (G10, G01)

        # Matrices from equation C = AP
        A = np.zeros((states_number, states_number))  # matrix A from the master equation of
        A[states_number, :] = 1  # normalize all probabilities to 1

        A[0, 0] = -rate_10
        A[0, 1] = rate_01

        C = np.zeros((states_number, 1))  # vector of solutions to rate equations and normalisation
        C[states_number, 0] = 1

        for n in range(2, states_number + 1):  # no reason to not include the state N since we start from 0 and go to N, not N-1
            state_energy = self.state_energy(n)
            energy_diff = state_energy - energy_list[n - 1]  # hence why we calculate state 0 outside the loop
            energy_list.append(state_energy)
            energy_diff_list.append(energy_diff)

            rate_sup_inf = self.RL * fermi((energy_diff - self.muL) * self.beta) + self.RR * fermi((energy_diff - self.muL) * self.beta)
            rate_inf_sup = self.RL * (1 - fermi((energy_diff - self.muL) * self.beta)) + self.RR * (1 - fermi((energy_diff - self.muL) * self.beta))

            rates_list.append((rate_sup_inf, rate_inf_sup))

            A[n - 1, n - 2] = - rates_list[n - 1][0]
            A[n - 1, n - 1] = - rates_list[n - 1][1] - rate_sup_inf
            A[n - 1, n + 2] = rate_inf_sup

        probability_vector = np.linalg.solve(A, C)

        return current_matrix

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
