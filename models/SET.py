import numpy as np
from numpy import ndarray
from typing import Tuple, Any

from constants import e, kB
from utils.fermi import fermi


class SET:
    def __init__(self, C1: float, C2: float, Vd: ndarray, Vs: float = 0.0, Vg: ndarray = 0, Cg: float = 0, Q0: float = 0.0, RL: float = 1.6e6, RR: float = 1.6e6, T: float = 0.1, alpha: float = 0.0):
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
        # self.Qext = self.Q0 + self.Cg * (self.Vg - self.alpha * self.V)

        # Charging energy
        self.Ec = e ** 2 / self.C_total
        # Chemical potential
        self.muR = -e * self.Vs  # right barrier (source)
        self.muL = -e * self.Vd  # left barrier (drain)

    def state_energy(self, N, Vd, Vg):
        """
        Calculate energy of state N
        :param N: Index of the state
        :param Vd: Fixed value of Vd
        :param Vg: Fixed value of Vg
        :return:
        """
        return (-e * (N - self.Q0 - 0.5) + self.C1 * self.Vs + self.C2 * Vd + self.Cg * Vg) ** 2 / (2 * e * self.C_total)

    def rate(self, energy_diff, mu, distribution: callable):
        """
        Calculate rate between two states
        :param energy_diff: Energy difference between states
        :param mu: chemical potential
        :param distribution: The distribution function, either fermi or 1 - fermi
        :return:
        """
        return self.RL * distribution((energy_diff - mu) * self.beta) + self.RR * distribution((energy_diff - mu) * self.beta)

    def current(self, states_number: int = 5) -> ndarray:
        """
        Calculate the current through the device for N states available
        :param states_number: Number of states to consider (N=5 by default)
        :return: Free energy, as a list of numpy array from E0 to E5 (or E_N for N set by states_number)
        """
        if states_number < 1:
            raise ValueError('Number of states cannot be smaller than 1.')

        Vd_2d, Vg_2d = np.meshgrid(self.Vd, self.Vg)
        # Initialise current matrix
        current_matrix = np.full((len(self.Vg), len(self.Vd)), e)

        # Matrices from equation C = AP
        A = np.zeros((states_number + 1, states_number + 1))  # matrix A from the master equation of
        A[states_number, :] = 1  # normalize all probabilities to 1 on last row

        C = np.zeros((states_number + 1, 1))  # vector of solutions to rate equations and normalisation
        C[states_number, 0] = 1

        # TODO: Try recursive function for this, might be interesting and more efficient (not sure)
        for i in range(len(self.Vg)):
            for j in range(len(self.Vd)):
                Vd = Vd_2d[i, j]
                Vg = Vg_2d[i, j]
                muL = - Vd_2d[i, j]

                # Set energy of states
                energies = [self.state_energy(n, Vd, Vg) for n in range(0, states_number + 1)]
                energy_diff_arr = np.diff(np.array(energies))

                # Set rates between states N and N+1
                rates_list = [(self.rate(energy_diff, muL, fermi), self.rate(energy_diff, muL, lambda E: 1 - fermi(E))) for energy_diff in energy_diff_arr]

                # Define the elements of matrix A of master equation
                A[0, 0] = - rates_list[0][0]  # into and out of N=0 state
                A[0, 1] = rates_list[0][1]
                A[states_number, :] = 1  # normalise all probabilities to 1
                for n in range(1, states_number):
                    A[n, n - 1] = rates_list[n - 1][0]
                    A[n, n] = - rates_list[n - 1][1] - rates_list[n][0]
                    A[n, n + 1] = rates_list[n][1]

                # Solve the matrix equation to get probability vector
                probability_vector = np.linalg.solve(A, C)

                for n in range(len(energy_diff_arr)):
                    on = probability_vector[n] * self.RR * fermi((energy_diff_arr[n] - self.muR) * self.beta)
                    off = probability_vector[n] * self.RR * (1 - fermi((energy_diff_arr[n] - self.muR) * self.beta))

                    current_matrix[i, j] += on
                    current_matrix[i, j] -= off

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
