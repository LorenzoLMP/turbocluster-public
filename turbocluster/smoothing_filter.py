import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling
from .spherical_tiling import SphericalTiling
import nvtx

from .smoothing_filter_kernels import *
from .generic_kernels import *
from .derivative_smooth_filt_kernels import *


class SmoothingFilter:
    """
    """

    def __init__(self, snap, center, widths, orientation=None, search_radius=None,
                 npix=128, threadsperblock=256, tilingType='cartesian', numPhi=-1, 
                 numTheta=-1, rMin=-1.0, rMax=-1.0, typeGrid='log', powerGrid=0,
                gauss_multiplier=4):
        """
        If spherical=True, npix is the number of intervals in the radial direction
        in the phi and theta direction we have npix, and npix/2 intervals
        by default
        """
        rng0 = nvtx.start_range(message="init_smoothing")
        
        
        if orientation is not None:
            raise RuntimeError('not implemented')
        if (tilingType == 'spherical'):
            if (rMin < 0.0) or (rMax < 0.0) or (rMax < rMin):
                raise RuntimeError('With spherical \
                you need to provide a non-negative \
                rMin and rMax > rMin')

        self.snap = snap

        if (tilingType == 'spherical'):
            self.spherical = True
            self.cartesian = False
        elif (tilingType == 'cartesian'):
            self.cartesian = True
            self.spherical = False

        # only used with spherical tiling
        if (typeGrid == 'log'):
            self.typeGrid = 0
        elif (typeGrid == 'power-law'):
            self.typeGrid = 1
        self.powerGrid = powerGrid
        
        code_length = self.snap.length

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

        if (tilingType == 'spherical'):
            numRad = npix
            
            if (numPhi < 0):
                numPhi = npix
            if (numTheta < 0):
                numTheta = int(npix/2)

            if hasattr(rMin, 'unit'):
                self.rMin = rMin.copy
                assert rMin.unit == code_length.unit, 'this restriction applies'
            elif pa.settings.use_units:
                self.rMin = rMin * code_length
            else:
                self.rMin = rMin

            if hasattr(rMax, 'unit'):
                self.rMax = rMax.copy
                assert rMax.unit == code_length.unit, 'this restriction applies'
            elif pa.settings.use_units:
                self.rMax = rMax * code_length
            else:
                self.rMax = rMax

        

        if orientation is None:
            self.orientation = None
        else:
            self.orientation = orientation.copy

        self.pos = self.snap["0_Coordinates"]

        # Calculate the smoothing length
        rng = nvtx.start_range(message="smoothing_length")
        ## this is the radius of the 'spherical' voronoi cell
        self.hsml = np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))
        nvtx.end_range(rng)

        if pa.settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        if search_radius is None:
            search_radius = 10.0 * self.hsml
        elif not hasattr(search_radius, 'unit'):
            search_radius = search_radius*code_length

        # self.multiplier = 4.0
        # self.multiplier = 6.0 ## this is better
        self.multiplier = gauss_multiplier

        if not isinstance(search_radius.value, np.ndarray):
            # it is not a vector already
            search_radius = np.ones(self.hsml.shape)*search_radius
        else:
            assert search_radius.shape[0] == self.hsml.shape[0]
            
        if pa.settings.use_units:
            # use units
            assert search_radius.unit == code_length.unit, 'this restriction applies'
            self.search_radius = 1.1*self.multiplier*search_radius
        else:
            # does not need units
            self.search_radius = 1.1*self.multiplier*np.array(search_radius)

        
        

        # rng = nvtx.start_range(message="region_selection")
        if (tilingType == 'cartesian'):
            self._do_region_selection()
        elif (tilingType == 'spherical'):
            self._do_region_selection_spherical()
        # nvtx.end_range(rng)

        self.extra_layer_thickness = np.max(self.hsml) + self.max_search_radius
        if pa.settings.use_units:
            self.extra_layer_thickness_value = self.extra_layer_thickness.value
        else:
            self.extra_layer_thickness_value = self.extra_layer_thickness

        # Create tiling
        if (tilingType == 'cartesian'):
            self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'],
                                        self.gpu_variables['widths'], self.extra_layer_thickness_value, npix=npix,
                                        threadsperblock=threadsperblock)
        elif (tilingType == 'spherical'):
            self.tile = SphericalTiling(self.gpu_variables['pos'], self.gpu_variables['center'],
                                        rMin, rMax, self.extra_layer_thickness_value,
                                        nRadial=numRad, nPhi=numPhi, nTheta=numTheta,
                                        type=typeGrid, power=powerGrid, threadsperblock=256)

        # Do the sorting
        self.gpu_variables['pos'] = self.gpu_variables['pos'][self.tile.sort_index, :]
        self.gpu_variables['hsml'] = self.gpu_variables['hsml'][self.tile.sort_index]

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock

        nvtx.end_range(rng0)

    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        thickness = self.hsml 
        rng = nvtx.start_range(message="region_selection")
        if self.orientation is None:
            get_index = pa.util.get_index_of_cubic_region_plus_thin_layer
            self.indicesFirstPass = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness,
                                   snap.box)
            self.max_search_radius = np.max(self.search_radius[self.indicesFirstPass])
            thickness = self.hsml + self.max_search_radius
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness,
                                   snap.box)
        else:
            get_index = pa.util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.indicesFirstPass = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness, snap.box,
                                   self.orientation)
            self.max_search_radius = np.max(self.search_radius[self.indicesFirstPass])
            thickness = self.hsml + self.max_search_radius
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness, snap.box,
                                   self.orientation)
        nvtx.end_range(rng)

        self.pos = self.pos[self.index]
        self.hsml = self.hsml[self.index]

        self._send_data_to_gpu()

    def _do_region_selection_spherical(self):
        """ 
        rMin, rMax are the domain computational boundaries
        chosen by the user
        _rMin, _rMax are the lower and upper limits of 
        the radial grid (computed by SphericalTiling)
        _rMin < rMin
        _rMax > rMax
        """

        center = self.center
        # widths = self.widths
        snap = self.snap
        rMin = self.rMin
        rMax = self.rMax

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        thickness = self.hsml         
        get_index = pa.util.get_index_of_radial_range_plus_thin_layer
        self.indicesFirstPass = get_index(self.snap["0_Coordinates"],
                               center, rMin, rMax, thickness)
        self.max_search_radius = np.max(self.search_radius[self.indicesFirstPass])
        thickness = self.hsml + self.max_search_radius
        self.index = get_index(self.snap["0_Coordinates"],
                               center, rMin, rMax, thickness)

        self.pos = self.pos[self.index]
        self.hsml = self.hsml[self.index]

        self._send_data_to_gpu()

    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if pa.settings.use_units:
            self.gpu_variables['pos'] = cp.array(self.pos.value)
            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
        else:
            self.gpu_variables['pos'] = cp.array(self.pos)
            self.gpu_variables['hsml'] = cp.array(self.hsml)

        if self.orientation is not None:
            self.gpu_variables['rotation_matrix'] = cp.array(
                self.orientation.rotation_matrix)

        if pa.settings.use_units:
            if self.cartesian: 
                self.gpu_variables['widths'] = cp.array(self.widths.value)
            elif self.spherical:
                self.gpu_variables['rMin'] = cp.array(self.rMin.value)
                self.gpu_variables['rMax'] = cp.array(self.rMax.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            if self.cartesian: 
                self.gpu_variables['widths'] = cp.array(self.widths)
            elif self.spherical:
                self.gpu_variables['rMin'] = cp.array(self.rMin)
                self.gpu_variables['rMax'] = cp.array(self.rMax)
            self.gpu_variables['center'] = cp.array(self.center)

    def _apply_filter_gpu(self, variable_str, weight, filter_type, iterative):
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        variable = self.gpu_variables[variable_str]
        center = self.gpu_variables['center']
        offsets = self.tile.off_sets
        
        max_search_radius = self.max_search_radius.value/self.multiplier


        if 'selection' in self.gpu_variables.keys():
            isParticleInSelection = self.gpu_variables['selection']
        else:
            isParticleInSelection = cp.ones(variable.shape,dtype="bool")
            # isParticleInSelection = None

        
        if self.cartesian:
            tile_widths = self.tile.tile_widths
            widths = self.gpu_variables['widths']
            npixs = self.tile.npixs
        elif self.spherical:
            _rMin = self.tile._rMin
            _rMax = self.tile._rMax
            rMin = self.rMin.value
            rMax = self.rMax.value
            nSects = self.tile.nSects
            spacings = self.tile.spacings
            typeGrid = self.typeGrid
            power    = self.powerGrid
        
        
        filter_lengths = self.gpu_variables['filter_lengths']
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1

        iterativeFilter = 0 # not iterative
        if iterative:
            iterativeFilter = 1 # iterative
            
        smooth_var = cp.zeros_like(variable)
        hitsNeighbours = cp.zeros(variable.shape,dtype="int")
        isParticleInDomain = cp.zeros(variable.shape,dtype="int")
        hasConverged = cp.zeros(variable.shape,dtype="int")
        numIterations = cp.zeros(variable.shape,dtype="int")
        filter_lengths_out = cp.zeros(variable.shape,dtype="float")

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        if self.cartesian:
            rng = nvtx.start_range(message="cartesian filter")
            apply_filter[self.blocks_1d, self.threadsperblock](pos, hsml, tile_index, start_index_for_tile,
                                                           particles_per_tile, tile_widths,
                                                           variable, weights, offsets, npixs, center, widths, 
                                                           filter_lengths, smooth_var, filter_type, hitsNeighbours,
                                                              isParticleInDomain, iterativeFilter, hasConverged, 
                                                               numIterations, filter_lengths_out, self.multiplier, max_search_radius, isParticleInSelection)
            nvtx.end_range(rng)
        elif self.spherical:
            rng = nvtx.start_range(message="spherical filter")
            apply_filter_spherical[self.blocks_1d, self.threadsperblock](pos, hsml, tile_index, start_index_for_tile,
                                                           particles_per_tile, spacings,
                                                           variable, weights, nSects, center, rMin, rMax, _rMin, _rMax,
                                                           filter_lengths, smooth_var, filter_type, hitsNeighbours,
                                                           isParticleInDomain, typeGrid, power, iterativeFilter,
                                                           hasConverged, numIterations, filter_lengths_out, self.multiplier, max_search_radius, isParticleInSelection)
            nvtx.end_range(rng)
        
        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        
        if iterative:
            self.filter_lengths_out = filter_lengths_out[self.tile.unsort_index]
            self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
            self.numIterationsUnSorted = numIterations[self.tile.unsort_index]
            tot_particles_domain = np.sum(self.isParticleInDomainUnSorted)
            num_part_converg = np.sum(self.hasConvergedUnSorted[self.isParticleInDomainUnSorted>0])
            percent_converg = num_part_converg/tot_particles_domain
            print("%.2f percent of particles (%d / %d) has converged"%(percent_converg*100,num_part_converg,tot_particles_domain))
            mean_iter = np.mean(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            std_iter = np.std(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            print("Number iterations needed: %.2f (+/- %.2f)"%(mean_iter,std_iter))
            print("Min/Max iterations needed: %d / %d"%(np.min(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0]),
                                                        np.max(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])))
        
            
        return cp.asnumpy(smooth_var[self.tile.unsort_index])

    def _send_variable_to_gpu(self, variable, gpu_key='input_variable'):
        if isinstance(variable, str):
            variable_str = str(variable)
            err_msg = 'filter only works on gas'
            assert int(variable[0]) == 0, err_msg
            variable = self.snap[variable]
        else:
            variable_str = gpu_key
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars can be filtered'

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

    def filter_variable(self, variable, filter_length, weight=None, filter_type="mean", iterative=False,
                       shared_mem=False, Nmax=64, optimized=False, selection=None):
        """
        shared_mem has been tested only with filter_type="mean"
        Nmax is the max number of particles per block. Each tile is split
        in "logic" blocks with Nmax particles max (can be less, but not zero)
        and assigned to exactly 1 block of threads with Nmax threads
        selection can be a subset of the cubic/spherical region
        """

        if (shared_mem and optimized):
            raise RuntimeError('shared_mem and optimized are incompatible')
        if (shared_mem and iterative):
            raise RuntimeError('shared_mem and iterative are incompatible')
            
        rng0 = nvtx.start_range(message="do_filter")
        
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            if isinstance(weight, str):
                self._send_variable_to_gpu(weight)
            else:
                raise RuntimeError('has to be a string')

        if not hasattr(filter_length, 'unit'):
            raise RuntimeError('filter_length must have unit')

        # send filter_length to gpu
        if isinstance(filter_length.value, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            if (self.multiplier*np.max(filter_length[self.indicesFirstPass]) > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            if (self.multiplier*filter_length > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length.value

        if selection is not None:
            self._send_variable_to_gpu(selection*self.snap.uq(''), gpu_key='selection')

        # Do the filtering
        if not shared_mem:
            if optimized:
                smooth_variable = self._apply_filter_gpu_optimized(variable_str, weight, filter_type, iterative)
            else: 
                smooth_variable = self._apply_filter_gpu(variable_str, weight, filter_type, iterative)
        else:
            smooth_variable = self._apply_filter_gpu_shared(variable_str, weight, filter_type, Nmax)

        if unit_quantity is not None:
            smooth_variable = smooth_variable * unit_quantity

        nvtx.end_range(rng0)
        
        return smooth_variable

    def _apply_filter_gpu_shared(self, variable_str, weight, filter_type, Nmax):
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        tile_widths = self.tile.tile_widths
        variable = self.gpu_variables[variable_str]
        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        filter_lengths = self.gpu_variables['filter_lengths']
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        elif filter_type == "mexican-hat":
            filter_type = 2
        
        smooth_var = cp.zeros_like(variable)

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        # now we do a bunch of preparatory operations to subdivide
        # the cartesian tiling into logical non-empty blocks
        # of Nmax particles
        self.tile.compactify_grid(Nmax)
        # number of block in compactified grid
        numBlocksCompGrid = self.tile.compactGrid.shape[0]
        # now we want to know which of the blocks actually fall inside the
        # filtering domain. This is the case if at least one particle within the block
        # is inside the filtering domain
        # the result is stored as 0 (no) or 1 (yes) in the following array
        # by the kernel check_block
        isBlockInDomain = cp.zeros(numBlocksCompGrid)
        blocks_1d = (numBlocksCompGrid + (self.threadsperblock - 1)) // self.threadsperblock
        check_block[blocks_1d, self.threadsperblock](pos, hsml, self.tile.compactGrid, center, widths, isBlockInDomain)
        # we compute the cumulative occupancy to know how many blocks
        # we need per tile and in total (numBlocksInDomain)
        cumulative_occupancy = cp.cumsum(isBlockInDomain)
        numBlocksInDomain = int(cumulative_occupancy[-1])
        # final preparatory step where we build the logical 
        # compactified grid of particles in the filtering domain
        compactGrid = cp.zeros((numBlocksInDomain,3),dtype=int)
        compactify_in_domain[blocks_1d, self.threadsperblock](self.tile.compactGrid, cumulative_occupancy, 
                                                              isBlockInDomain, compactGrid)
        # now we are ready to launch the filtering kernel
        self.numBlocksInDomain = numBlocksInDomain
        self.compactGrid = compactGrid
        self.isBlockInDomain = isBlockInDomain
        # we need to define how much shared memory per block of threads
        # we are going to need (in bytes)
        sharedMemBuf = 8 * Nmax * 12
        self.sharedMemBuf = sharedMemBuf
        print("numBlocksInDomain = %d"%(numBlocksInDomain))

        hitsNeighbours = cp.zeros(variable.shape,dtype="int")
        isParticleInDomain = cp.zeros(variable.shape,dtype="int")

        rng = nvtx.start_range(message="shared cartesian filter")
        apply_filter_shared[numBlocksInDomain, Nmax, 0, sharedMemBuf](compactGrid, pos, hsml, tile_index, 
                                                     start_index_for_tile, particles_per_tile, 
                                                     tile_widths, variable, weights, center, 
                                                     widths, npixs, filter_lengths, smooth_var, filter_type,
                                                                     hitsNeighbours, isParticleInDomain, self.multiplier)
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        return cp.asnumpy(smooth_var[self.tile.unsort_index])


    def _apply_filter_gpu_optimized(self, variable_str, weight, filter_type, iterative):
        """
        The idea behind this 'optimized' version is to check if the particle
        is in the domain _beforehand_, and then run the filtering kernel 
        only on those that are in the domain. There is a certain speedup 
        in doing so (for small-size problems running time can be 1/3)
        I have also improved the tile searching within the kernel: now only
        the tiles that *overlap* with the filtering radius of each particle
        are selected, without wasting time looping over those that do not
        For now I am adding it as an option to the filter_variable function
        (optimized=True) to allow a comparison with the baseline 
        (optimized=False)
        """

        if self.spherical:
            raise RuntimeError('optimized filter has only \
                                been tested with cartesian grids')
            
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # - self.tile.off_sets[None,:]
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        tile_widths = self.tile.tile_widths
        variable = self.gpu_variables[variable_str]
        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        offsets = self.tile.off_sets
        filter_lengths = self.gpu_variables['filter_lengths']

        max_search_radius = self.max_search_radius.value/self.multiplier

        
        
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        elif filter_type == "mexican-hat":
            filter_type = 2

        iterativeFilter = 0 # not iterative
        if iterative:
            iterativeFilter = 1 # iterative
        

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        isParticleInDomain = cp.zeros(pos.shape[0])
        
        check_particle[self.blocks_1d, self.threadsperblock](pos, hsml, center, widths, isParticleInDomain)

        if 'selection' in self.gpu_variables.keys():
            ## if we want to filter only a selection of the domain
            isParticleInSelection = self.gpu_variables['selection']
            isParticleInDomain *= isParticleInSelection
                   
        self.isParticleInDomain = isParticleInDomain
        cumulative_occupancy = cp.cumsum(isParticleInDomain)
        numParticlesInDomain = int(cumulative_occupancy[-1])
        oldIndex = cp.zeros(numParticlesInDomain,dtype=int)
        
        compactify_particles[self.blocks_1d, self.threadsperblock](pos, tile_index,
                                        cumulative_occupancy.flatten(), isParticleInDomain, 
                                        oldIndex)
        self.oldIndex = oldIndex
        
        blocks_1d = (numParticlesInDomain + (self.threadsperblock - 1)) // self.threadsperblock
        
        smooth_var = cp.zeros_like(variable)
        hitsNeighbours = cp.zeros(variable.shape,dtype="int")
        # isParticleInDomain = cp.zeros(variable.shape,dtype="int")
        hasConverged = cp.zeros(variable.shape,dtype="int")
        numIterations = cp.zeros(variable.shape,dtype="int")
        filter_lengths_out = cp.zeros(variable.shape,dtype="float")

        rng = nvtx.start_range(message="cartesian filter (optimized)")
        apply_filter_optimized[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, 
                                                          start_index_for_tile, particles_per_tile, tile_widths,
                                                           variable, weights, offsets, npixs, center, widths, 
                                                          filter_lengths, smooth_var, filter_type, hitsNeighbours,
                                                              isParticleInDomain, iterativeFilter, hasConverged, 
                                                               numIterations, filter_lengths_out, self.multiplier, max_search_radius)
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
        
        if iterative:
            self.filter_lengths_out = filter_lengths_out[self.tile.unsort_index]
            self.numIterationsUnSorted = numIterations[self.tile.unsort_index]
            tot_particles_domain = np.sum(self.isParticleInDomainUnSorted)
            num_part_converg = np.sum(self.hasConvergedUnSorted[self.isParticleInDomainUnSorted>0])
            percent_converg = num_part_converg/tot_particles_domain
            print("%.2f percent of particles (%d / %d) has converged"%(percent_converg*100,num_part_converg,tot_particles_domain))
            mean_iter = np.mean(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            std_iter = np.std(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            print("Number iterations needed: %.2f (+/- %.2f)"%(mean_iter,std_iter))
            print("Min/Max iterations needed: %d / %d"%(np.min(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0]),
                                                        np.max(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])))
        
        
        return cp.asnumpy(smooth_var[self.tile.unsort_index])

    def filter_vector(self, variable_x, variable_y, variable_z, filter_length, weight=None, 
                       filter_type="mean", iterative=False,
                       shared_mem=False, Nmax=64, optimized=False, selection=None):
        """
        shared_mem has been tested only with filter_type="mean"
        Nmax is the max number of particles per block. Each tile is split
        in "logic" blocks with Nmax particles max (can be less, but not zero)
        and assigned to exactly 1 block of threads with Nmax threads
        """

        if (shared_mem and optimized):
            raise RuntimeError('shared_mem and optimized are incompatible')
        if (shared_mem and iterative):
            raise RuntimeError('shared_mem and iterative are incompatible')
            
        rng0 = nvtx.start_range(message="do_filter")
        
        variable_x_str, unit_quantity = self._send_variable_to_gpu(variable_x, gpu_key='input_variable_x')
        variable_y_str, unit_quantity = self._send_variable_to_gpu(variable_y, gpu_key='input_variable_y')
        variable_z_str, unit_quantity = self._send_variable_to_gpu(variable_z, gpu_key='input_variable_z')

        if weight is not None:
            if isinstance(weight, str):
                self._send_variable_to_gpu(weight)
            else:
                raise RuntimeError('has to be a string')

        if not hasattr(filter_length, 'unit'):
            raise RuntimeError('filter_length must have unit')

        # send filter_length to gpu
        if isinstance(filter_length.value, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            if (np.max(self.multiplier*filter_length[self.indicesFirstPass]) > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            if (self.multiplier*filter_length > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length.value

        if selection is not None:
            self._send_variable_to_gpu(selection*self.snap.uq(''), gpu_key='selection')

        # Do the filtering
        if not shared_mem:
            if optimized:
                # if (norm == 1):
                #     smooth_variable_x, smooth_variable_y, smooth_variable_z  = self._apply_filter_gpu_optimized_vector1(
                #                                                 variable_x_str, variable_y_str,
                #                                                variable_z_str, weight, 
                #                                                filter_type, iterative)
                # elif (norm == 2):
                smooth_variable_x, smooth_variable_y, smooth_variable_z  = self._apply_filter_gpu_optimized_vector(
                                                            variable_x_str, variable_y_str,
                                                           variable_z_str, weight, 
                                                           filter_type, iterative)
        #     else: 
        #         smooth_variable = self._apply_filter_gpu(variable_str, weight, filter_type, iterative)
        # else:
        #     smooth_variable = self._apply_filter_gpu_shared(variable_str, weight, filter_type, Nmax)

        if unit_quantity is not None:
            smooth_variable_x = smooth_variable_x * unit_quantity
            smooth_variable_y = smooth_variable_y * unit_quantity
            smooth_variable_z = smooth_variable_z * unit_quantity

        nvtx.end_range(rng0)
        
        return smooth_variable_x, smooth_variable_y, smooth_variable_z

    def _apply_filter_gpu_optimized_vector(self, variable_x_str, variable_y_str, variable_z_str, weight, filter_type, iterative):
        """
        The idea behind this 'optimized' version is to check if the particle
        is in the domain _beforehand_, and then run the filtering kernel 
        only on those that are in the domain. There is a certain speedup 
        in doing so (for small-size problems running time can be 1/3)
        I have also improved the tile searching within the kernel: now only
        the tiles that *overlap* with the filtering radius of each particle
        are selected, without wasting time looping over those that do not
        For now I am adding it as an option to the filter_variable function
        (optimized=True) to allow a comparison with the baseline 
        (optimized=False)
        """

        if self.spherical:
            raise RuntimeError('optimized filter has only \
                                been tested with cartesian grids')
            
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # - self.tile.off_sets[None,:]
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        tile_widths = self.tile.tile_widths

        max_search_radius = self.max_search_radius.value/self.multiplier
        
        variable_x = self.gpu_variables[variable_x_str]
        variable_y = self.gpu_variables[variable_y_str]
        variable_z = self.gpu_variables[variable_z_str]
        
        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        offsets = self.tile.off_sets
        filter_lengths = self.gpu_variables['filter_lengths']
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        elif filter_type == "mexican-hat":
            filter_type = 2

        iterativeFilter = 0 # not iterative
        if iterative:
            iterativeFilter = 1 # iterative
        

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable_x)

        isParticleInDomain = cp.zeros(pos.shape[0])
        
        check_particle[self.blocks_1d, self.threadsperblock](pos, hsml, center, widths, isParticleInDomain)

        if 'selection' in self.gpu_variables.keys():
            ## if we want to filter only a selection of the domain
            isParticleInSelection = self.gpu_variables['selection']
            isParticleInDomain *= isParticleInSelection
            
        self.isParticleInDomain = isParticleInDomain
        cumulative_occupancy = cp.cumsum(isParticleInDomain)
        numParticlesInDomain = int(cumulative_occupancy[-1])
        oldIndex = cp.zeros(numParticlesInDomain,dtype=int)
        
        compactify_particles[self.blocks_1d, self.threadsperblock](pos, tile_index,
                                        cumulative_occupancy.flatten(), isParticleInDomain, 
                                        oldIndex)
        self.oldIndex = oldIndex
        
        blocks_1d = (numParticlesInDomain + (self.threadsperblock - 1)) // self.threadsperblock
        
        smooth_var_x = cp.zeros_like(variable_x)
        smooth_var_y = cp.zeros_like(variable_y)
        smooth_var_z = cp.zeros_like(variable_z)
        
        hitsNeighbours = cp.zeros(variable_x.shape,dtype="int")
        # isParticleInDomain = cp.zeros(variable.shape,dtype="int")
        hasConverged = cp.zeros(variable_x.shape,dtype="int")
        numIterations = cp.zeros(variable_x.shape,dtype="int")
        filter_lengths_out = cp.zeros(variable_x.shape,dtype="float")

        rng = nvtx.start_range(message="cartesian filter (optimized)")
        apply_filter_optimized_vector[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, 
                                                          start_index_for_tile, particles_per_tile, tile_widths,
                                                           variable_x, variable_y, variable_z, weights, offsets, npixs, center, widths, 
                                                          filter_lengths, smooth_var_x, smooth_var_y, smooth_var_z, filter_type, hitsNeighbours,
                                                              isParticleInDomain, iterativeFilter, hasConverged, 
                                                               numIterations, filter_lengths_out, self.multiplier, max_search_radius)
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        
        if iterative:
            self.filter_lengths_out = filter_lengths_out[self.tile.unsort_index]
            self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
            self.numIterationsUnSorted = numIterations[self.tile.unsort_index]
            tot_particles_domain = np.sum(self.isParticleInDomainUnSorted)
            num_part_converg = np.sum(self.hasConvergedUnSorted[self.isParticleInDomainUnSorted>0])
            percent_converg = num_part_converg/tot_particles_domain
            print("%.2f percent of particles (%d / %d) has converged"%(percent_converg*100,num_part_converg,tot_particles_domain))
            mean_iter = np.mean(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            std_iter = np.std(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])
            print("Number iterations needed: %.2f (+/- %.2f)"%(mean_iter,std_iter))
            print("Min/Max iterations needed: %d / %d"%(np.min(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0]),
                                                        np.max(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])))
        
        
        return cp.asnumpy(smooth_var_x[self.tile.unsort_index]), cp.asnumpy(smooth_var_y[self.tile.unsort_index]), cp.asnumpy(smooth_var_z[self.tile.unsort_index]) 


    def derivative_variable(self, variable, filter_length, weight=None, filter_type="gaussian", 
                            iterative=False, optimized=True, selection=None):
        """
        This function computes the derivative of a smoothed variable
        by moving the derivative under the integral sign and doing a convolution
        with the derivative of the kernel. If the integration bounds are 
        extended to infinity this does not introduce extra boundary terms
        (see Leibniz' integral rule https://en.wikipedia.org/wiki/Leibniz_integral_rule)
        and for sufficiently fast-decaying smoothing kernels the error in 
        truncating the integration bounds should be small anyway.
        """

        if (filter_type != "gaussian"):
            raise RuntimeError('derivative currently has only gaussian')
        if (iterative):
            raise RuntimeError('derivative has no iterative option')
        if (not optimized):
            raise RuntimeError('derivative only uses optimized kernel')
        if (weight is not None):
            print('Derivative does not support weighted integration. Argument will be ignored')
            
        rng0 = nvtx.start_range(message="do_filter")
        
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if not hasattr(filter_length, 'unit'):
            raise RuntimeError('filter_length must have unit')
        filt_length_unit = filter_length.unit_quantity

        # send filter_length to gpu
        if isinstance(filter_length.value, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            if (self.multiplier*np.max(filter_length[self.indicesFirstPass]) > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            if (self.multiplier*filter_length > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length.value

        if selection is not None:
            self._send_variable_to_gpu(selection*self.snap.uq(''), gpu_key='selection')

        # Compute the gradient in x, y, z
        # grad_var is a 3-dim vector
        grad_var = self._apply_derivative_gpu(variable_str, filter_type)
        
        if unit_quantity is not None:
            grad_var = grad_var * unit_quantity / filt_length_unit

        nvtx.end_range(rng0)
        
        return grad_var

    def derivative_vector(self, variable_x, variable_y, variable_z, filter_length, 
                            weight=None, filter_type="gaussian", 
                            iterative=False, optimized=True, selection=None):
        """
        This function computes the derivative of a smoothed variable
        by moving the derivative under the integral sign and doing a convolution
        with the derivative of the kernel. If the integration bounds are 
        extended to infinity this does not introduce extra boundary terms
        (see Leibniz' integral rule https://en.wikipedia.org/wiki/Leibniz_integral_rule)
        and for sufficiently fast-decaying smoothing kernels the error in 
        truncating the integration bounds should be small anyway.
        """

        if (filter_type != "gaussian"):
            raise RuntimeError('derivative currently has only gaussian')
        if (iterative):
            raise RuntimeError('derivative has no iterative option')
        if (not optimized):
            raise RuntimeError('derivative only uses optimized kernel')
        if (weight is not None):
            print('Derivative does not support weighted integration. Argument will be ignored')
            
        rng0 = nvtx.start_range(message="do_filter")

        variable_x_str, unit_quantity = self._send_variable_to_gpu(variable_x, gpu_key='input_variable_x')
        variable_y_str, unit_quantity = self._send_variable_to_gpu(variable_y, gpu_key='input_variable_y')
        variable_z_str, unit_quantity = self._send_variable_to_gpu(variable_z, gpu_key='input_variable_z')

        if not hasattr(filter_length, 'unit'):
            raise RuntimeError('filter_length must have unit')
        filt_length_unit = filter_length.unit_quantity

        # send filter_length to gpu
        if isinstance(filter_length.value, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            if (self.multiplier*np.max(filter_length[self.indicesFirstPass]) > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            if (self.multiplier*filter_length > self.max_search_radius):
                raise RuntimeError('The chosen filter length (x4) is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length.value

        if selection is not None:
            self._send_variable_to_gpu(selection*self.snap.uq(''), gpu_key='selection')

        # Compute the gradient in x, y, z
        # grad_var_x is a 3-dim vector that contains 
        # the x, y, and z derivative of var_x
        # same for the others
        grad_var_x, grad_var_y, grad_var_z = self._apply_derivative_vector_gpu(variable_x_str, 
                                                                              variable_y_str,
                                                                              variable_z_str, 
                                                                              filter_type)
        
        if unit_quantity is not None:
            grad_var_x = grad_var_x * unit_quantity / filt_length_unit
            grad_var_y = grad_var_y * unit_quantity / filt_length_unit
            grad_var_z = grad_var_z * unit_quantity / filt_length_unit

        nvtx.end_range(rng0)
        
        return grad_var_x, grad_var_y, grad_var_z

    def _apply_derivative_gpu(self, variable_str, filter_type):
        """
        
        """

        if self.spherical:
            raise RuntimeError('optimized filter has only \
                                been tested with cartesian grids')

        if (filter_type != "gaussian"):
            raise RuntimeError('derivative currently has only gaussian')
            
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # - self.tile.off_sets[None,:]
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        tile_widths = self.tile.tile_widths
        variable = self.gpu_variables[variable_str]
        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        offsets = self.tile.off_sets
        filter_lengths = self.gpu_variables['filter_lengths']

        max_search_radius = self.max_search_radius.value/self.multiplier
        
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        elif filter_type == "mexican-hat":
            filter_type = 2
       

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        isParticleInDomain = cp.zeros(pos.shape[0])
        
        check_particle[self.blocks_1d, self.threadsperblock](pos, hsml, center, widths, isParticleInDomain)

        if 'selection' in self.gpu_variables.keys():
            ## if we want to filter only a selection of the domain
            isParticleInSelection = self.gpu_variables['selection']
            isParticleInDomain *= isParticleInSelection
                   
        self.isParticleInDomain = isParticleInDomain
        cumulative_occupancy = cp.cumsum(isParticleInDomain)
        numParticlesInDomain = int(cumulative_occupancy[-1])
        oldIndex = cp.zeros(numParticlesInDomain,dtype=int)
        
        compactify_particles[self.blocks_1d, self.threadsperblock](pos, tile_index,
                                        cumulative_occupancy.flatten(), isParticleInDomain, 
                                        oldIndex)
        self.oldIndex = oldIndex
        
        blocks_1d = (numParticlesInDomain + (self.threadsperblock - 1)) // self.threadsperblock
        
        grad_x_var = cp.zeros_like(variable)
        grad_y_var = cp.zeros_like(variable)
        grad_z_var = cp.zeros_like(variable)
        
        hitsNeighbours = cp.zeros(variable.shape,dtype="int")
        hasConverged = cp.zeros(variable.shape,dtype="int")

        rng = nvtx.start_range(message="cartesian filter (optimized)")

        apply_derivative[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, 
                     start_index_for_tile, particles_per_tile, tile_widths,
                     variable, offsets, npixs, center, widths, filter_lengths, 
                     grad_x_var, grad_y_var, grad_z_var, filter_type, hitsNeighbours, isParticleInDomain, 
                     hasConverged, self.multiplier)
        
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
        
        grad_x_var_unsort = cp.asnumpy(grad_x_var[self.tile.unsort_index])
        grad_y_var_unsort = cp.asnumpy(grad_y_var[self.tile.unsort_index])
        grad_z_var_unsort = cp.asnumpy(grad_z_var[self.tile.unsort_index])

        grad_var = np.stack((grad_x_var_unsort, grad_y_var_unsort, grad_z_var_unsort), axis=1)
        
        return grad_var 

    def _apply_derivative_vector_gpu(self, variable_x_str, variable_y_str, variable_z_str, 
                                    filter_type):
        """
        
        """

        if self.spherical:
            raise RuntimeError('optimized filter has only \
                                been tested with cartesian grids')

        if (filter_type != "gaussian"):
            raise RuntimeError('derivative currently has only gaussian')
            
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # - self.tile.off_sets[None,:]
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        tile_widths = self.tile.tile_widths

        variable_x = self.gpu_variables[variable_x_str]
        variable_y = self.gpu_variables[variable_y_str]
        variable_z = self.gpu_variables[variable_z_str]
        
        npixs = self.tile.npixs
        center = self.gpu_variables['center']
        widths = self.gpu_variables['widths']
        offsets = self.tile.off_sets
        filter_lengths = self.gpu_variables['filter_lengths']

        max_search_radius = self.max_search_radius.value/self.multiplier
        
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        elif filter_type == "mexican-hat":
            filter_type = 2
       

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        isParticleInDomain = cp.zeros(pos.shape[0])
        
        check_particle[self.blocks_1d, self.threadsperblock](pos, hsml, center, widths, isParticleInDomain)

        if 'selection' in self.gpu_variables.keys():
            ## if we want to filter only a selection of the domain
            isParticleInSelection = self.gpu_variables['selection']
            isParticleInDomain *= isParticleInSelection
                   
        self.isParticleInDomain = isParticleInDomain
        cumulative_occupancy = cp.cumsum(isParticleInDomain)
        numParticlesInDomain = int(cumulative_occupancy[-1])
        oldIndex = cp.zeros(numParticlesInDomain,dtype=int)
        
        compactify_particles[self.blocks_1d, self.threadsperblock](pos, tile_index,
                                        cumulative_occupancy.flatten(), isParticleInDomain, 
                                        oldIndex)
        self.oldIndex = oldIndex
        
        blocks_1d = (numParticlesInDomain + (self.threadsperblock - 1)) // self.threadsperblock
        
        grad_x_vec_x = cp.zeros_like(variable_x)
        grad_x_vec_y = cp.zeros_like(variable_x)
        grad_x_vec_z = cp.zeros_like(variable_x)

        grad_y_vec_x = cp.zeros_like(variable_x)
        grad_y_vec_y = cp.zeros_like(variable_x)
        grad_y_vec_z = cp.zeros_like(variable_x)

        grad_z_vec_x = cp.zeros_like(variable_x)
        grad_z_vec_y = cp.zeros_like(variable_x)
        grad_z_vec_z = cp.zeros_like(variable_x)
        
        hitsNeighbours = cp.zeros(variable_x.shape,dtype="int")
        hasConverged = cp.zeros(variable_x.shape,dtype="int")

        rng = nvtx.start_range(message="cartesian filter (optimized)")

        apply_derivative_vector[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, 
                     start_index_for_tile, particles_per_tile, tile_widths,
                     variable_x, variable_y, variable_z, offsets, npixs, 
                     center, widths, filter_lengths, 
                     grad_x_vec_x, grad_x_vec_y, grad_x_vec_z, 
                     grad_y_vec_x, grad_y_vec_y, grad_y_vec_z,
                     grad_z_vec_x, grad_z_vec_y, grad_z_vec_z, filter_type, hitsNeighbours, 
                     isParticleInDomain, hasConverged, self.multiplier)
        
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
        
        grad_x_vec_x_unsort = cp.asnumpy(grad_x_vec_x[self.tile.unsort_index])
        grad_y_vec_x_unsort = cp.asnumpy(grad_y_vec_x[self.tile.unsort_index])
        grad_z_vec_x_unsort = cp.asnumpy(grad_z_vec_x[self.tile.unsort_index])

        grad_vec_x = np.stack((grad_x_vec_x_unsort, grad_y_vec_x_unsort, grad_z_vec_x_unsort), axis=1)

        grad_x_vec_y_unsort = cp.asnumpy(grad_x_vec_y[self.tile.unsort_index])
        grad_y_vec_y_unsort = cp.asnumpy(grad_y_vec_y[self.tile.unsort_index])
        grad_z_vec_y_unsort = cp.asnumpy(grad_z_vec_y[self.tile.unsort_index])

        grad_vec_y = np.stack((grad_x_vec_y_unsort, grad_y_vec_y_unsort, grad_z_vec_y_unsort), axis=1)

        grad_x_vec_z_unsort = cp.asnumpy(grad_x_vec_z[self.tile.unsort_index])
        grad_y_vec_z_unsort = cp.asnumpy(grad_y_vec_z[self.tile.unsort_index])
        grad_z_vec_z_unsort = cp.asnumpy(grad_z_vec_z[self.tile.unsort_index])

        grad_vec_z = np.stack((grad_x_vec_z_unsort, grad_y_vec_z_unsort, grad_z_vec_z_unsort), axis=1)
        
        return grad_vec_x, grad_vec_y, grad_vec_z



    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        self.release_gpu_memory()

    def release_gpu_memory(self):
        # TODO: Add deletion of all GPU variables stored in self
        if hasattr(self, 'gpu_variables'):
            for key in list(self.gpu_variables):
                del self.gpu_variables[key]
            del self.gpu_variables
        if hasattr(self, 'tile'):
            self.tile.release_gpu_memory()
            del self.tile

        # cp._default_memory_pool.free_all_blocks()
