import numpy as np
import cupy as cp
from numba import cuda
import nvtx
import paicos as pa

from ..data_init import DataGpuInit
from ..CudaKernels.generic_kernels import *
from ..CudaKernels.deposit_kernels import *
from ..CudaKernels.power_spectra_kernels import *
from ..cartesian_tiling import CartesianTiling


class DepositCartesianGrid(DataGpuInit):
    """
    """
    def __init__(self, snap, center, widths, orientation=None, npix=128, threadsperblock=256, **kwargs):
        
        super().__init__(snap, center, widths, orientation=orientation, threadsperblock=threadsperblock)

        self.__dict__.update(kwargs)

        if 'kernel_type' not in self.__dict__:
            raise ValueError("Please provide kernel type. Possible options are: NGP, CIC, TSC, PCS.")
        kernel_type = self.__dict__['kernel_type']

        # if 'npoints' not in self.__dict__:
        #     raise ValueError("Please provide number of gridpoints for the deposition.")
        # npoints = self.__dict__['npoints']

        self.npix = npix

        if 'pos' not in self.__dict__:
            print("No `pos' argument given. Defaults to gas particles")
            self.pos = self.snap["0_Coordinates"]
        else:
            self.pos = self.__dict__['pos']

        if 'hsml' not in self.__dict__:
            print("No `hsml' argument given. Defaults to gas particles")
            # Calculate the smoothing length
            ## this is the radius of the 'spherical' voronoi cell
            # test with 4 times radius 
            self.hsml = 4.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))
        else:
            self.hsml = self.__dict__['hsml']

        if pa.settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        if kernel_type == "NGP":
            self.support = 0
            # raise RuntimeError('Deposition with kernel NGP \
            #     has issues')
        elif kernel_type == "CIC":
            self.support = 1
        elif kernel_type == "TSC":
            self.support = 2
        elif kernel_type == "PCS":
            self.support = 3

        if (self.cartesian):
            thickness = self.support*self.hsml 
            self.index = self._do_region_selection(thickness, self.pos)

        self._send_variable_to_gpu(self.pos, gpu_key='pos')
        self._send_variable_to_gpu(self.hsml, gpu_key='hsml')

        self._send_variable_to_gpu(self.widths, gpu_key='widths')
        self._send_variable_to_gpu(self.center, gpu_key='center')

        self._rotate_coordinates()

        self.extra_layer_thickness = self.support*np.max(self.hsml[self.index])
        if pa.settings.use_units:
            self.extra_layer_thickness = self.extra_layer_thickness.value

        # Create tiling
        if (self.cartesian):
            self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'], self.gpu_variables['widths'], 0.0, npix=npix, threadsperblock=threadsperblock)


        self.npoints = self.tile.npixs + 1

        
        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (self.threadsperblock - 1)) // self.threadsperblock

    def deposit_variable(self, variable, weight=None):
        """
        
        """
            
        rng0 = nvtx.start_range(message="do_deposition")
        
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            self._send_variable_to_gpu(weight, gpu_key='weight')

        deposited_variable = self._do_deposition_gpu(variable_str, weight)

        if unit_quantity is not None:
            deposited_variable = deposited_variable * unit_quantity

        nvtx.end_range(rng0)
        
        return deposited_variable

    def _do_deposition_gpu(self, variable_str, weight):
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # tile_index = self.tile.tile_index
        # start_index_for_tile = self.tile.start_index_for_tile
        # particles_per_tile = self.tile.particles_per_tile
        variable = self.gpu_variables[variable_str]
        center = self.gpu_variables['center']
        offsets = self.tile.off_sets
        
        if self.cartesian:
            tile_widths = self.tile.tile_widths
            widths = self.gpu_variables['widths']
            npixs = self.tile.npixs
        
        kernel_type = self.support

        deposited_var = cp.zeros(self.npoints.tolist(), dtype="float")
        scratch = cp.zeros(self.npoints.tolist(), dtype="float")

        if weight is not None:
            weights = self.gpu_variables['weight']
        else:
            weights = cp.ones_like(pos.shape[0])

        rng = nvtx.start_range(message="cartesian deposition")

        if (len(variable.shape) > 1):
            # is a vector
            # TODO: 
            pass
        else:
            # is a scalar
            deposit_on_grid[self.blocks_1d, self.threadsperblock](pos, hsml, tile_widths,
                            variable, weights, offsets, npixs, center, widths, deposited_var, 
                            scratch, kernel_type)
        nvtx.end_range(rng)

        # self.scratch = scratch
        # self.deposited_var = deposited_var
        
        # return cp.asnumpy(deposited_var/scratch)
        if (np.argwhere(deposited_var==0.0).size > 0):
            print("Warning: %d grid points have zero values"%(np.argwhere(deposited_var==0.0).size))
        
        # return cp.asnumpy(cp.where(scratch>0,deposited_var/scratch,0.0))
        return cp.asnumpy(deposited_var)

    

        