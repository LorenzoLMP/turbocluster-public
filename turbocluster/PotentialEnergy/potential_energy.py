import numpy as np
import cupy as cp
from numba import cuda
import math
import paicos as pa
import nvtx

from ..data_init import DataGpuInit
from ..cartesian_tiling import CartesianTiling
from ..CudaKernels.potential_energy_kernels import *


class PotentialEnergy(DataGpuInit):
    """
    This class computes the potential energy of a region by tiling the particles in a 
    uniform cartesian grid, and approximating long-range interactions as two point masses
    """

    def __init__(self, snap, center, widths, orientation=None, npix=128, threadsperblock=256, **kwargs):
        
        super().__init__(snap, center, widths, orientation=orientation, npix=npix, threadsperblock=threadsperblock)

    # def _prepare_data(self):

        self.__dict__.update(kwargs)
    # def _prepare_data(self):

        if 'mass' not in self.__dict__:
            raise ValueError("Please provide masses")

        mass = self.__dict__['mass']
        self.code_mass = code_mass = mass.uq

        if 'smoothing_length' not in self.__dict__:
            raise ValueError("Please provide smoothing_length")
        smoothing_length = self.__dict__['smoothing_length']

        if hasattr(smoothing_length, 'unit'):
            assert smoothing_length.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            smoothing_length = smoothing_length * code_length
            # raise RuntimeError('smoothing_length must have unit')

        if isinstance(smoothing_length.value, np.ndarray):
            assert smoothing_length.shape[0] == self.index.shape[0]

        if hasattr(mass, 'unit'):
            assert mass.unit == code_mass.unit, 'this restriction applies'
        elif pa.settings.use_units:
            mass = mass * code_mass
            # raise RuntimeError('smoothing_length must have unit')

        if isinstance(smoothing_length.value, np.ndarray):
            assert smoothing_length.shape[0] == self.pos.shape[0]
        else:
            smoothing_length = np.ones(self.pos.shape[0]) * smoothing_length.value

        if isinstance(mass.value, np.ndarray):
            assert mass.shape[0] == self.pos.shape[0]
        else:
            mass = np.ones(self.pos.shape[0]) * mass.value

        if (self.cartesian):
            thickness = np.zeros(self.pos.shape[0])*code_length
            self.index = self._do_region_selection(thickness)

        ## requires selection of the region
        self._send_variable_to_gpu(self.pos, gpu_key='pos')
        self._send_variable_to_gpu(self.widths, gpu_key='widths')
        self._send_variable_to_gpu(self.center, gpu_key='center')

        self._send_variable_to_gpu(mass, gpu_key='mass')
        self._send_variable_to_gpu(smoothing_length, gpu_key='smoothing_length')

        self._rotate_coordinates()

        # Create tiling
        if (self.cartesian):
            self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'], self.gpu_variables['widths'], 0.0, npix=npix,threadsperblock=threadsperblock)

        # Do the sorting
        for variable_str in self.gpu_variables:
            if self.gpu_variables[variable_str].shape[0] == self.tile.sort_index.shape[0]:
                self.gpu_variables[variable_str] = self.gpu_variables[variable_str][
                    self.tile.sort_index]

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (self.threadsperblock - 1)) // self.threadsperblock

        # compute total mass in each tile
        self.tile.mass_per_tile = self.tile.accumulate_per_tile(self.gpu_variables['mass'])
        

        # finds max smoothing length in each tile
        self.tile.max_hsml_per_tile = self.tile.findmax_per_tile(self.gpu_variables['smoothing_length'])

        nvtx.end_range(rng0)

        self.G = G = pa.astropy.constants.G

    def compute_potential(self, angle=15):
        """
        angle is in degrees
        smoothing_length is an array of the smoothing lengths
        for the gravitational potential (or a constant scalar)
        """
            
        rng0 = nvtx.start_range(message="do_computation potential")
        
        # variable_str, unit_quantity = self._send_variable_to_gpu(variable)


        # Do the computation
        potential = self._compute_potential_gpu(angle)

        potential = cp.asnumpy(potential)

        if pa.settings.use_units:
            potential = self.G * potential * self.code_mass**2 / self.code_length

        nvtx.end_range(rng0)
        
        return potential

    def _compute_potential_gpu(self, angle):
        """
        """
        pos = self.gpu_variables['pos']
        mass = self.gpu_variables['mass']
        smoothing_length = self.gpu_variables['smoothing_length']
        
        
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        max_hsml_per_tile = self.tile.max_hsml_per_tile
        tile_widths = self.tile.tile_widths
        mass_per_tile = self.tile.mass_per_tile
        

        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        offsets = self.tile.off_sets
        tan_theta0 = np.tan(0.5*angle*np.pi/180 + 1e-12) ## angle is in degrees, theta0 in radiants
        ## theta0 is half the angle of the isosceles triangle with short side H and orthogonal D
        ## i.e. tan(theta0) = (H/2)/D

        potential_in_tile = cp.zeros(particles_per_tile.shape, dtype="double")
        self.particles_hit = particles_hit = cp.zeros(pos.shape[0], dtype='int')
        self.tiles_hit = tiles_hit = cp.zeros(particles_per_tile.shape, dtype='int')

        rng = nvtx.start_range(message="potential energy kernel")

        threadsperblock = self.threadsperblock
        numBlocks = int(npixs[0]*npixs[1]*npixs[2])
        blocks_1d = (numBlocks + (threadsperblock - 1)) // threadsperblock

        ## this kernel is called with a thread assigned to each block
        ## of the cartesian grid, NOT particle!
        compute_potential_energy_coarse[blocks_1d, threadsperblock](pos, mass, smoothing_length,tile_index, start_index_for_tile,particles_per_tile, mass_per_tile,max_hsml_per_tile, tile_widths, tan_theta0,offsets, npixs, center, widths, potential_in_tile, tiles_hit)

        blocks_1d = (pos.shape[0] + (threadsperblock - 1)) // threadsperblock
        ## this kernel is called with a thread assigned to each particle!
        compute_potential_energy_N2[blocks_1d, threadsperblock](pos, mass, smoothing_length,tile_index, start_index_for_tile,particles_per_tile, mass_per_tile,max_hsml_per_tile, tile_widths, tan_theta0,offsets, npixs, center, widths,potential_in_tile, particles_hit)
        
        nvtx.end_range(rng)

        potential = 0.5*cp.sum(potential_in_tile)

        return potential

    

