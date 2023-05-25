import numpy as np
import matplotlib.pyplot as plt

from models.SET import SET

# Parameters
nbr_points_Vg = 500
nbr_points_V = 200

Q0 = 0  # background charge

limit = 0.5e-3  # voltage range

Vg = np.linspace(start=-0.03, stop=0.1, num=nbr_points_Vg)  # external potential
V = np.linspace(start=-0.0025, stop=0.0025, num=nbr_points_V)  # gate potential

C1 = 1.98e-17  # capacitance source (F)
C2 = 4.80e-17  # capacitance drain (F)
Cg = 5.09e-18  # gate capacitance (F)

circuit = SET(C1=C1, C2=C2, Vd=V, Vg=Vg, Cg=Cg, T=0.3)
I = circuit.current(5)

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.figure()

plt.pcolormesh(Vg, V*100, (I*1e12).T, cmap=plt.get_cmap('seismic'))
clb = plt.colorbar()
clb.ax.set_title('$I (pA)$', fontsize=12)
clb.ax.tick_params(labelsize=12)

# plt.xlim(-0.02, 0.1)
# plt.ylim(V[0]*100, V[len(V) - 1]*100)

plt.ylabel('$V_D$ (mV)', fontsize=12)
plt.xlabel('$V_g$ (V)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
