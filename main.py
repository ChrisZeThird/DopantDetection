import numpy as np
import matplotlib.pyplot as plt

from models.SET import SET


# Parameters
nbr_points = 200

Q0 = 0  # background charge

limit = 0.5e-3  # voltage range
V = np.linspace(start=-limit, stop=limit, num=nbr_points)  # external potential
Vg = np.linspace(start=-limit, stop=limit, num=nbr_points)  # gate potential

C1 = 0.87e-18  # capacitance source (F)
C2 = 0.87e-18  # capacitance drain (F)
Cg = 3.52e-18  # gate capacitance (F)

circuit = SET(C1, C2, V, Vg, Cg)

N1, N2 = 1, 1
helmholtz_energy = circuit.free_energy(N1, N2)

delta_free_energy_plus_N1, delta_free_energy_minus_N1, delta_free_energy_plus_N2, delta_free_energy_minus_N2 = circuit.energy_diff(N1, N2)

# Apply condition of Coulomb blockade
stability_diagram = np.zeros((nbr_points, nbr_points))
indices = (delta_free_energy_plus_N1 < 0) & (delta_free_energy_minus_N1 < 0) & (delta_free_energy_plus_N2 < 0) & (delta_free_energy_minus_N2 < 0)
stability_diagram[indices] = 1

# Displaying stability diagram
plt.pcolormesh(V, Vg, stability_diagram, cmap='copper')
plt.xlabel('V')
plt.ylabel('Vg')
plt.title('SET Stability Diagram')
plt.show()
