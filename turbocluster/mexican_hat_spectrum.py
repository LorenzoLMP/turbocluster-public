import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling
from .spherical_tiling import SphericalTiling
from .smoothing_filter import SmoothingFilter
import nvtx

from .smoothing_filter_kernels import *
from .generic_kernels import *
from .helper_functions import *


class MexicanHatPowerSpectrum(SmoothingFilter):
    """
    """
    def compute_spectrum(self, variable, mask=None, weight=None, optimized=False):

        snap = self.snap
        
        if isinstance(variable, str):
            var_unit = snap[variable].uq
            # var_shape = snap[variable].shape
        else:
            var_unit = variable.uq
            # var_shape  = variable.shape

        

        # self.kmin = np.sqrt(2.0)/(self.max_search_radius.value/4.1)
        max_filter_length = np.max(self.widths)/5.
        if (max_filter_length > self.max_search_radius / self.multiplier):
            raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            
        self.kmin = np.sqrt(2.0)/max_filter_length.value
        self.kmax = 12.*self.kmin
        k_vec = np.logspace(np.log10(self.kmin), np.log10(self.kmax), 12)/self.snap.length
        # this is a volume integral of the variance
        var_variance = np.zeros(k_vec.shape)*var_unit**2*self.snap.length**3 
        

        for i in range(len(k_vec)):
            k = k_vec[i]
            filt_len = (np.sqrt(2.0)/k)

            if (mask is None):
                var_filtered, _ = extract_turbulent_scalar(snap, self, 
                                                          variable, filt_len, 
                                                          weight, test_type="diff_of_gaussians",
                                                           filter_type="mexican-hat",
                                                           iterative=False)
                
                var_variance[i] = volume_integral(snap, self, var_filtered**2,
                                              self.indicesFirstPass)
            else:
                ## equation A9 Arevalo+2012
                G_sigma1_var, _ = extract_turbulent_scalar(snap, self, 
                                                          variable, filt_len/np.sqrt(1.0+epsilon), 
                                                          weight, filter_type="gaussian",
                                                           iterative=False)
                # note that it does not matter what unit is mask
                # because in the end it cancels out
                # but we need a unit for the following function to work

                # we use the convention that mask=1=True on the cells which we want
                # to include, and mask=0=False on the cells to exclude
                G_sigma1_mask, _ = extract_turbulent_scalar(snap, self, 
                                                          mask*var_unit, filt_len/np.sqrt(1.0+epsilon), 
                                                          weight, filter_type="gaussian",
                                                           iterative=False)

                G_sigma2_var, _ = extract_turbulent_scalar(snap, self, 
                                                          variable, filt_len*np.sqrt(1.0+epsilon), 
                                                          weight, filter_type="gaussian",
                                                           iterative=False)
                G_sigma2_mask, _ = extract_turbulent_scalar(snap, self, 
                                                          mask*var_unit, filt_len*np.sqrt(1.0+epsilon), 
                                                          weight, filter_type="gaussian",
                                                           iterative=False)

                S_k = (G_sigma1_var / G_sigma1_mask - G_sigma2_var / G_sigma2_mask ) * mask * var_unit

                ## this is necessary to get rid of the NaNs
                ## due to division by G_sigma1/2_mask 
                ## (which is zero on a subset of the masked cells)
                S_k[~mask] = 0.0
                
                S_k /= epsilon

                vol_tot = volume_integral(snap, self, np.ones(mask.shape),
                                              self.indicesFirstPass)

                vol_non_masked = volume_integral(snap, self, mask,
                                              self.indicesFirstPass)

                # equation A10 of Arevalo+2012
                var_variance[i] = (vol_tot/vol_non_masked) * volume_integral(snap, self, 
                                                                          S_k**2,
                                                                          self.indicesFirstPass)
        # equation A11 of Arevalo+2012 replacing k_r -> k / (2 \pi) 

        ## this is the 3D energy spectral density
        power_spectrum3D = (var_variance/k_vec**3)*(np.pi**(3./2.)*2**(11./2.))/(3.*5./2.)

        ## for 1D energy spectral density
        ## (we are assuming S_3D does not depend on the angular variable)
        power_spectrum1D = power_spectrum3D * 4.0*np.pi*(k_vec/(2.*np.pi))**2
            
        
        return power_spectrum1D, k_vec