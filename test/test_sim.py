import matplotlib.pyplot as plt
from time import time
import numpy as np
from csimulate import simulate_spheres
import chard_sphere as cs


N = 500
r_max = 20
l_max = 512

l = (l_max / r_max / 2)

system = cs.HSMC(N, [l, l, l], [False, False, False])
system.fill_hs()
print(system)

positions = system.get_positions()  * r_max * 2

radii = np.random.uniform(20, 20, N)
intensities = np.random.uniform(0.5, 5, N)

t0 = time()
sim = simulate_spheres(
    positions.T, intensities, radii,
    l_max, l_max, l_max
)
print(f"new: {time() - t0}s")


plt.imshow(sim[:, :, l_max//2])
plt.show()
