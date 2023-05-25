import numpy as np
import matplotlib.pyplot as plt

from models.SET import SET


# Parameters
nbr_points = 200

Q0 = 0  # background charge

limit = 0.5e-3  # voltage range
Vd = np.linspace(start=-limit, stop=limit, num=nbr_points)  # external potential
Vg = np.linspace(start=-limit, stop=limit, num=nbr_points)  # gate potential

Vd, Vg = np.meshgrid(Vd, Vg)

C1 = 0.87e-18  # capacitance source (F)
C2 = 0.87e-18  # capacitance drain (F)
Cg = 3.52e-18  # gate capacitance (F)

circuit = SET(C1, C2, Vd, Vg=Vg, Cg=Cg)
I = circuit.current()

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.figure()
plt.pcolormesh(Vg, Vd*100, I*1e12, cmap=plt.get_cmap('seismic'))
clb = plt.colorbar()
clb.ax.set_title('$I (pA)$', fontsize=12)
clb.ax.tick_params(labelsize=12)

plt.xlim(-0.02, 0.1)
plt.ylabel('$V_D$ (mV)', fontsize=12)
plt.xlabel('$V_G$ (V)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.show()