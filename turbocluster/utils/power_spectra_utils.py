import numpy as np
import cupy as cp
from numba import cuda
import math
import nvtx
import paicos as pa

from ..cartesian_tiling import CartesianTiling
from ..SmoothingFilter.smoothing_filter import SmoothingFilter


from ..CudaKernels.smoothing_filter_kernels import *
from ..CudaKernels.power_spectra_kernels import *
from ..CudaKernels.generic_kernels import *
from ..helper_functions import *


def power_spectrum1d(depo, deposited_variable, **kwargs):
    """
    kwargs:
        window : 1D window generation function
           The window function should accept one argument: the window length.
           Example: window=scipy.signal.windows.hann
    """

    Nx, Ny, Nz = deposited_variable.shape

    if pa.settings.use_units:
        # if hasattr(self.widths, 'unit'):
        Lx, Ly, Lz = depo.widths.value
        L_unit = depo.widths.uq
    else:
        Lx, Ly, Lz = depo.widths

    if pa.settings.use_units:
        depo_variable = deposited_variable.value.copy()
        variable_unit = deposited_variable.uq
    else:
        depo_variable = deposited_variable.copy()

    voxel_real_space = (Lx / Nx) * (Ly / Ny) * (Lz / Nz)
    energy_real_space = np.sum(depo_variable**2 * voxel_real_space)
    print('energy (real space) = %.4e' % (energy_real_space))

    # this is if we want to do windowing
    if 'window' in kwargs:
        depo_variable, ndim_window = nd_window(depo_variable,
                                               kwargs["window"])

    # Send variable to gpu
    d_depo_variable = cp.array(depo_variable)

    hat_depo_variable = cp.fft.rfftn(d_depo_variable, s=(Nx, Ny, Nz))
    Ntotalcomplex = Nx * Ny * Nz

    # create the wavevectors
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    kz = 2.0 * np.pi * np.fft.rfftfreq(Nz, d=Lz / Nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    K2 = KX**2 + KY**2 + KZ**2

    kvec = np.sqrt(KX**2 + KY**2 + KZ**2)

    kxmax = (2.0 * np.pi / Lx) * (Nx // 2)
    kymax = (2.0 * np.pi / Ly) * (Ny // 2)
    kzmax = (2.0 * np.pi / Lz) * (Nz // 2)

    kmax = np.sqrt(kxmax**2 + kymax**2 + kzmax**2)
    # I take the coarsest grid in k-space
    deltak = 2.0 * np.pi / np.min([Lx, Ly, Lz])

    nbin = int(kmax / deltak + 0.5)
    n1d = np.arange(0, nbin)
    k1d = deltak * n1d

    # these are now the *wavenumbers*
    ## (not wavevectors)
    wavenum = kvec / deltak
    # store them on device
    d_wavenum = cp.array(wavenum)

    d_powerspectr = cp.zeros(k1d.shape)
    blocks_1d = (Ntotalcomplex + (depo.threadsperblock - 1)
                 ) // depo.threadsperblock

    # launch kernel
    gpu_power_spectrum1d[blocks_1d, depo.threadsperblock](hat_depo_variable, d_wavenum,
                                                          (Nx, Ny, Nz),
                                                          d_powerspectr)

    powerspectr = cp.asnumpy(d_powerspectr) * (Lx * Ly * Lz) * \
        (2.0 * np.pi / deltak)  # for energy/frequency
    # powerspectr = cp.asnumpy(d_powerspectr)*(2.0*np.pi/deltak)    ## for energy/volume/frequency

    if 'window' in kwargs:  # for normalization with window function
        # factor from https://holometer.fnal.gov/GH_FFT.pdf Eq. 21
        # powerspectr /= np.square(np.sum(ndim_window))/(np.sum(np.square(ndim_window))*(Nx*Ny*Nz))
        powerspectr /= np.sum(np.square(ndim_window)) / (Nx * Ny * Nz)

    energy_fourier_space = np.sum(powerspectr * deltak / (2.0 * np.pi))
    print('energy (fourier space) = %.4e' % (energy_fourier_space))

    if pa.settings.use_units:
        k1d = k1d / L_unit
        # (for energy/frequency: 3 powers for Lx*Ly*Lz + 1 power for 1/deltak)
        powerspectr = powerspectr * L_unit**4 * variable_unit**2

    return powerspectr, k1d, (KX, KY, KZ, cp.asnumpy(hat_depo_variable))


def mexicanhatPowerSpectrum(snap, sf, variable, max_filter_length=None, mask=None, weight=None, optimized=False):

    # snap = self.snap

    if isinstance(variable, str):
        variable = snap[variable]
    var_unit = variable.uq

    # if len(variable.shape) == 1 is a scalar
    # if len(variable.shape) == 2 is a vector
    ndims = 1
    if (len(variable.shape) > 1):
        ndims = variable.shape[-1]

    # self.kmin = np.sqrt(2.0)/(self.max_search_radius.value/4.1)
    if (max_filter_length == None):
        max_filter_length = np.max(sf.widths) / 7.
    if (max_filter_length > sf.max_search_radius / sf.multiplier):
        raise RuntimeError('The chosen filter length (x4) is larger than the \
            maximum search radius. This would cause searching for cells that \
            have not been moved to the GPU. To solve this decrease \
            the filter length or increase the search radius accordingly')

    kmin = np.sqrt(2.0) / max_filter_length.value
    kmax = 15. * kmin
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), 15) / snap.length
    # this is a volume integral of the variance
    var_variance = np.zeros((len(k_vec), ndims)) * var_unit**2 * snap.length**3

    for i in range(len(k_vec)):
        k = k_vec[i]
        filt_len = (np.sqrt(2.0) / k)

        if (mask is None):
            var_filtered, _ = extract_turbulent_vector(
                snap, sf, variable, filt_len, weight, filter_type="mexican-hat", iterative=False)
            for n in range(ndims):
                var_variance[i, n] = volume_integral(
                    snap, var_filtered[:, n]**2, sf.indicesFirstPass)
        else:
            # equation A9 Arevalo+2012
            G_sigma1_var, _ = extract_turbulent_var(snap, sf, variable, filt_len / np.sqrt(
                1.0 + epsilon), weight, filter_type="gaussian", iterative=False)
            # note that it does not matter what unit is mask
            # because in the end it cancels out
            # but we need a unit for the following function to work

            # we use the convention that mask=1=True on the cells which we want
            # to include, and mask=0=False on the cells to exclude
            G_sigma1_mask, _ = extract_turbulent_var(snap, sf, mask * var_unit, filt_len / np.sqrt(
                1.0 + epsilon), weight, filter_type="gaussian", iterative=False)

            G_sigma2_var, _ = extract_turbulent_var(snap, sf, variable, filt_len * np.sqrt(
                1.0 + epsilon), weight, filter_type="gaussian", iterative=False)
            G_sigma2_mask, _ = extract_turbulent_var(snap, sf, mask * var_unit, filt_len * np.sqrt(
                1.0 + epsilon), weight, filter_type="gaussian", iterative=False)

            for n in range(ndims):

                S_k = (G_sigma1_var[:, n] / G_sigma1_mask
                       - G_sigma2_var[:, n] / G_sigma2_mask) * mask * var_unit

                # this is necessary to get rid of the NaNs
                # due to division by G_sigma1/2_mask
                # (which is zero on a subset of the masked cells)
                S_k[~mask] = 0.0

                S_k /= epsilon

                vol_tot = volume_integral(snap, np.ones(
                    mask.shape), sf.indicesFirstPass)

                vol_non_masked = volume_integral(
                    snap, mask, sf.indicesFirstPass)

                # equation A10 of Arevalo+2012
                var_variance[i, n] = (
                    vol_tot / vol_non_masked) * volume_integral(snap, S_k**2, sf.indicesFirstPass)

    # equation A11 of Arevalo+2012 replacing k_r -> k / (2 \pi)

    # this is the 3D energy spectral density
    power_spectrum3D = np.zeros((len(k_vec), ndims)) * \
        var_variance.uq * sf.snap.length**3

    for n in range(ndims):
        power_spectrum3D[:, n] = (var_variance[:, n] / k_vec**3) * \
            (np.pi**(3. / 2.) * 2**(11. / 2.)) / (3. * 5. / 2.)

    # for 1D energy spectral density
    # (we are assuming S_3D does not depend on the angular variable)
    power_spectrum1D = np.zeros((len(k_vec), ndims)) * \
        var_variance.uq * sf.snap.length

    for n in range(ndims):
        power_spectrum1D[:, n] = power_spectrum3D[n] * \
            4.0 * np.pi * (k_vec / (2. * np.pi))**2

    return power_spectrum1D, k_vec
