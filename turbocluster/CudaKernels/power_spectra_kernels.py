import cupy as cp
from numba import cuda
import scipy.signal
import numpy as np


def nd_window(data, filter_function, **kwargs):
    """
    https://stackoverflow.com/questions/27345861/extending-1d-function-across-3-dimensions-for-data-windowing
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.windows.hann
    """
    if hasattr(data, "unit"):
        windowed_data = data.value.copy()
    else:
        windowed_data = data.copy()
    ndim_window = np.ones(data.shape)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [
            1,
        ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        # np.power(window, (1.0/data.ndim), out=window)
        # window = window**(1.0/data.ndim)
        windowed_data *= window
        ndim_window *= window

    # norm = np.sqrt(np.sum(ndim_window**2)/ndim_window.size)
    # print('norm = ', norm)
    # windowed_data /= norm

    if hasattr(data, "unit"):
        windowed_data *= data.unit_quantity

    return windowed_data, ndim_window


@cuda.jit(inline=True)
def gpu_power_spectrum1d(vhat, wavenum, Ngrid, powerspectr):
    # vhat has shape:
    # (Nx,Ny,Nz) if complex transform, or
    # (Nx,Ny,Nz//2+1) if real transform
    # Ngrid is a tuple with the dimension
    # of the real grid (Nx, Ny, Nz)
    # Ncomplex is a tuple with the dimension
    # of the complex grid:
    # (Nx, Ny, Nz) if C2C
    # (Nx, Ny, Nz//2+1) if R2C
    # type = 0 for real fft
    # type = 1 for complex fft

    Nx, Ny, Nz = Ngrid
    ntotal = Nx * Ny * Nz
    ncomplex_kx, ncomplex_ky, ncomplex_kz = vhat.shape
    ntotal_complex = ncomplex_kx * ncomplex_ky * ncomplex_kz

    ip = cuda.grid(1)

    if ip < ntotal_complex:
        k = ip % ncomplex_kz
        ip_tmp = int((ip - k) / ncomplex_kz)
        j = ip_tmp % ncomplex_ky
        i = ip_tmp // ncomplex_ky

        power_at_freq = (vhat[i, j, k] * vhat[i, j, k].conjugate()).real

        # this is more properly the *wavenumber*
        # (not wavevector), i.e. it says which frequency
        # bin it belongs to
        freq = int(wavenum[i, j, k] + 0.5)
        # # if we are doing a real fft
        # # we need to double to take into account
        # # energy contained in the negative KZ midplane
        if k > 0:
            power_at_freq *= 2.0

        # for energy spectral density
        cuda.atomic.add(powerspectr, (freq), power_at_freq / (ntotal**2))
