"""
FINUFFT
Here I provide a first try at getting FINUFFT to work. The first step is to install finufft on Tycho,
as far as I recall I simply did

pip install finufft
"""

import finufft
import paicos as pa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Load a snapshot and select spherical region
# snap = pa.Snapshot('/lustre/astro/berlok/zoom-simulations-new-ics/halo_0003/adiabatic-mhd/zoom4_ics_v1/output', 247)
snap = pa.Snapshot(pa.data_dir, 247)
center = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0]
r_max = 1.0 * R200c
index = pa.util.get_index_of_radial_range(snap['0_Coordinates'], center, 0., r_max)

snap = snap.select(index, parttype=0)


# A simple example,
# https://finufft.readthedocs.io/en/latest/python.html#quick-start-examples
# https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan


k1 = 1.0 * 2 * np.pi
k2 = 0.0 * 2 * np.pi
k3 = 0.0 * 2 * np.pi

grid_size = 10

# the nonuniform points (rescaled to -1 to 1), should be -pi to pi?
x = (snap['0_Coordinates'][:, 0] - center[0]).value / R200c.value
y = (snap['0_Coordinates'][:, 1] - center[1]).value / R200c.value
z = (snap['0_Coordinates'][:, 2] - center[2]).value / R200c.value

# # their complex strengths
c = snap['0_Density'].value + 0.0j
# c = np.exp(1j*k1*x + 1j*k2*y + 1j*k3*z).real + 0.0j

# # desired number of Fourier modes

kx = np.arange(0, grid_size) * 2 * np.pi
ky = np.arange(0, grid_size) * 2 * np.pi
kz = np.arange(0, grid_size) * 2 * np.pi

# kx = np.logspace(-2, 2, grid_size)*2*np.pi
# ky = np.logspace(-2, 2, grid_size)*2*np.pi
# kz = np.logspace(-2, 2, grid_size)*2*np.pi

kxx, kyy, kzz = np.meshgrid(kx, ky, kz)

kx = kxx.flatten()
ky = kyy.flatten()
kz = kzz.flatten()


# calculate the NUFFT
f = finufft.nufft3d3(x, y, z, c, kx, ky, kz)
k = np.sqrt(kx**2 + ky**2 + kz**2)
k_index = np.argsort(k)
k = k[k_index]
f = f[k_index]

plt.figure(1)
plt.clf()
plt.loglog(k, np.abs(f))
plt.show()
