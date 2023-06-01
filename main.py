import numpy as np
import matplotlib.pyplot as plt

from models.SET import SET
from models.DoubleDot import DoubleDot

# SINGLE QUANTUM DOT

# # Parameters
# nbr_points_Vg = 500
# nbr_points_V = 200
#
# Q0 = 0  # background charge
#
# limit = 0.5e-3  # voltage range
#
# Vg = np.linspace(start=-0.03, stop=0.1, num=nbr_points_Vg)  # external potential
# V = np.linspace(start=-0.0025, stop=0.0025, num=nbr_points_V)  # gate potential
#
# C1 = 1.98e-17  # capacitance source (F)
# C2 = 4.80e-17  # capacitance drain (F)
# Cg = 5.09e-18  # gate capacitance (F)
#
# # E_QM = np.array([0, 0.4, 0.7, 0.8]) * 1e-3
#
# circuit = SET(C1=C1, C2=C2, Vd=V, Vg=Vg, Cg=Cg, T=0.1)
# I = circuit.current(5)
#
# plt.rc('text', usetex=False)
# plt.rc('font', family='serif')
# plt.figure()
#
# plt.pcolormesh(Vg, V*100, (I*1e12).T, cmap=plt.get_cmap('seismic'))
# clb = plt.colorbar()
# clb.ax.set_title('$I (pA)$', fontsize=12)
# clb.ax.tick_params(labelsize=12)
#
# # plt.xlim(-0.02, 0.1)
# # plt.ylim(V[0]*100, V[len(V) - 1]*100)
#
# plt.ylabel('$V_D$ (mV)', fontsize=12)
# plt.xlabel('$V_g$ (V)', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# plt.show()


# DOUBLE QUANTUM DOT

nbr_points_Vg1 = 500
nbr_points_Vg2 = 500

Vg1 = np.linspace(0, 0.05, nbr_points_Vg1)
Vg2 = np.linspace(0, 0.05, nbr_points_Vg2)

Vg1_2d, Vg2_2d = np.meshgrid(Vg1, Vg2)

Cg = 10.3e-18
T = 0.1

Cg1 = Cg
Cg2 = Cg
Cm = 0
CL = 0.5*Cg
CR = 0.5*Cg

Cm_limit = 0.8 * (CL + Cg1 + CR)

N1, N2 = 5, 5

DQD = DoubleDot(Vg1=Vg1_2d, Cg1=Cg1, Vg2=Vg2_2d, Cg2=Cg2, Cm=Cm, CL=CL, CR=CR, T=T)
print(DQD.EC1, DQD.EC2, DQD.ECm)
statistics_electron = DQD.electron_statistic(N1, N2)

# Stability diagram
plt.figure()
plt.pcolormesh(Vg1_2d, Vg2_2d, statistics_electron, cmap="seismic", shading="auto")
plt.xlabel("Vg1 (V)")
plt.ylabel("Vg2 (V)")
cbar = plt.colorbar()
cbar.set_label("Nb of electrons", rotation=90)

plt.show()
