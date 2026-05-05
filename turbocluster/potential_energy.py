import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling
import nvtx

from .potential_energy_kernels import *
# from .generic_kernels import *

class PotentialEnergy:
    """
    This class computes the potential energy of a region by tiling the particles in a 
    uniform cartesian grid, and approximating long-range interactions as two point masses
    """

    def __init__(self, snap, center, widths, pos, mass, smoothing_length, 
                 orientation=None, search_radius=None,
                 npix=128, threadsperblock=256, tilingType='cartesian'):

        """
        only cartesian tiling for now
        """
        rng0 = nvtx.start_range(message="init_potential_energy")

        if orientation is not None:
            raise RuntimeError('not implemented')
        self.orientation = None

        self.snap = snap

        self.cartesian = True

        self.code_length = code_length = pos.uq
        self.code_mass = code_mass = mass.uq

        if hasattr(center, 'unit'):
            self.center = center.copy
            assert center.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            self.center = np.array(center) * code_length
        else:
            self.center = np.array(center)

        if hasattr(widths, 'unit'):
            self.widths = widths.copy
            assert widths.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            self.widths = np.array(widths) * code_length
        else:
            self.widths = np.array(widths)

        if hasattr(smoothing_length, 'unit'):
            assert smoothing_length.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            smoothing_length = smoothing_length * code_length
            # raise RuntimeError('smoothing_length must have unit')

        if hasattr(mass, 'unit'):
            assert mass.unit == code_mass.unit, 'this restriction applies'
        elif pa.settings.use_units:
            mass = mass * code_mass
            # raise RuntimeError('smoothing_length must have unit')

        self.pos = pos

        if (tilingType == 'cartesian'):
            self._do_region_selection()

        # Create tiling
        if (tilingType == 'cartesian'):
            self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'],
                                        self.gpu_variables['widths'], 0.0, npix=npix,
                                        threadsperblock=threadsperblock)

            
        
        
        # Do the sorting
        self.gpu_variables['pos'] = self.gpu_variables['pos'][self.tile.sort_index, :]
        


        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock


        # send mass to gpu
        if isinstance(mass.value, np.ndarray):
            assert mass.shape[0] == self.index.shape[0]
            self._send_variable_to_gpu(mass, gpu_key='mass')
        else:
            self.gpu_variables['mass'] = cp.ones(self.Np) * mass.value

        # compute total mass in each tile
        self.tile.mass_per_tile = self.tile.accumulate_per_tile(self.gpu_variables['mass'])
        

        # send smoothing_length to gpu
        if isinstance(smoothing_length.value, np.ndarray):
            assert smoothing_length.shape[0] == self.index.shape[0]
            self._send_variable_to_gpu(smoothing_length, gpu_key='smoothing_length')
        else:
            self.gpu_variables['smoothing_length'] = cp.ones(self.Np) * smoothing_length.value

        # finds max smoothing length in each tile
        self.tile.max_hsml_per_tile = self.tile.findmax_per_tile(self.gpu_variables['smoothing_length'])


        nvtx.end_range(rng0)

        self.G = G = pa.astropy.constants.G


    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        # thickness = self.hsml 
        rng = nvtx.start_range(message="region_selection")
        if self.orientation is None:
            
            get_index = pa.util.get_index_of_cubic_region
            #### also here we are defaulting to gas particles!!!
            self.index = get_index(self.pos,
                                   center, widths, snap.box)
            
        else:
            get_index = pa.util.get_index_of_rotated_cubic_region
            self.index = get_index(self.pos,
                                   center, widths, snap.box,
                                   self.orientation)
        nvtx.end_range(rng)

        self.pos = self.pos[self.index]

        self._send_data_to_gpu()

    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if pa.settings.use_units:
            self.gpu_variables['pos'] = cp.array(self.pos.value)
        else:
            self.gpu_variables['pos'] = cp.array(self.pos)

        if self.orientation is not None:
            self.gpu_variables['rotation_matrix'] = cp.array(
                self.orientation.rotation_matrix)

        if pa.settings.use_units:
            if self.cartesian: 
                self.gpu_variables['widths'] = cp.array(self.widths.value)
            
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            if self.cartesian: 
                self.gpu_variables['widths'] = cp.array(self.widths)
            
            self.gpu_variables['center'] = cp.array(self.center)

    def _send_variable_to_gpu(self, variable, gpu_key='input_variable'):
        if isinstance(variable, str):
            variable_str = str(variable)
            variable = self.snap[variable]
        else:
            variable_str = gpu_key
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars'

        variable = variable[self.index]

        if variable_str in self.gpu_variables and variable_str != gpu_key:
            pass
        else:
            # Send variable to gpu
            if pa.settings.use_units:
                self.gpu_variables[variable_str] = cp.array(variable.value)
            else:
                self.gpu_variables[variable_str] = cp.array(variable)

            # Sort the variable according to tiling sorting
            self.gpu_variables[variable_str] = self.gpu_variables[variable_str][
                self.tile.sort_index]

        if isinstance(variable, pa.units.PaicosQuantity):
            unit_quantity = variable.unit_quantity
        else:
            unit_quantity = None

        return variable_str, unit_quantity

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
        compute_potential_energy_coarse[blocks_1d, threadsperblock](pos, mass, smoothing_length,
                                                             tile_index, start_index_for_tile,
                                                            particles_per_tile, mass_per_tile,
                                                            max_hsml_per_tile, tile_widths, tan_theta0,
                                                            offsets, npixs, center, widths, 
                                                            potential_in_tile, tiles_hit)

        blocks_1d = (pos.shape[0] + (threadsperblock - 1)) // threadsperblock
        ## this kernel is called with a thread assigned to each particle!
        compute_potential_energy_N2[blocks_1d, threadsperblock](pos, mass, smoothing_length,
                                                             tile_index, start_index_for_tile,
                                                            particles_per_tile, mass_per_tile,
                                                            max_hsml_per_tile, tile_widths, tan_theta0,
                                                            offsets, npixs, center, widths, 
                                                            potential_in_tile, particles_hit)
        
        nvtx.end_range(rng)

        
        potential = 0.5*cp.sum(potential_in_tile)

        return potential

    

