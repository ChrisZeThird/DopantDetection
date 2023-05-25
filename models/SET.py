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
        # Chemical potentials
        self.muR = -e * self.Vs  # chemical pot of right lead (source)
        self.muL = -e * self.Vd  # chemical potential of left lead (drain)

    def state_energy(self, N):
        """
        Calculate energy of state N
        :param N: Index of the state, can be int or array
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
        # Calculate energies of states
        energies = self.state_energy(np.arange(states_number))

        # Calculate rate matrices
        G_up = self.RL * fermi((energies[:-1] - energies[1:] - self.muR) * self.beta) + self.RR * fermi((energies[:-1] - energies[1:] - self.muR) * self.beta)
        G_down = self.RL * (1 - fermi((energies[:-1] - energies[1:] - self.muL) * self.beta)) + self.RR * (
                    1 - fermi((energies[:-1] - energies[1:] - self.muR) * self.beta))

        # Construct transition matrices
        G = np.diag(G_up, k=1) + np.diag(G_down, k=-1)

        # Set up normalization equation
        G[-1] = 1.0
        # Solve the matrix equation to get probability vector
        P = np.linalg.solve(G, np.zeros(self.nstates))

        # Calculate the source current by looking at transfer across right barrier
        on = P[:-1] * self.RR * fermi((energies[:-1] - energies[1:] - self.muR) * self.beta)
        off = P[1:] * self.RR * (1 - fermi((energies[:-1] - energies[1:] - self.muR) * self.beta))
        current_matrix = e * (np.sum(on) - np.sum(off))

        return current_matrix
