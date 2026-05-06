import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling
import nvtx

from .smoothing_filter_kernels import *
from .generic_kernels import *
from .derivative_smooth_filt_kernels import *


class SmoothingFilter:
    """
    """

    def __init__(self, snap, center, widths, orientation=None, search_radius=None,
                 npix=128, threadsperblock=256, tilingType='cartesian', gauss_multiplier=4):
        """
        
        """
        rng0 = nvtx.start_range(message="init_smoothing")
        

        self.snap = snap

        self.cartesian = True

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

        if not isinstance(search_radius.value, np.ndarray):
            # it is not a vector already
            search_radius = np.ones(self.hsml.shape)*search_radius
        else:
            assert search_radius.shape[0] == self.hsml.shape[0]

        self.multiplier = gauss_multiplier

        if pa.settings.use_units:
            # use units
            assert search_radius.unit == code_length.unit, 'this restriction applies'
            self.search_radius = 1.1*self.multiplier*search_radius
        else:
            # does not need units
            self.search_radius = 1.1*self.multiplier*np.array(search_radius)


        ## This selects the region and sends
        ## a bunch of arrays to the gpu
        # rng = nvtx.start_range(message="region_selection")
        if (tilingType == 'cartesian'):
            self._do_region_selection()
       

        self.extra_layer_thickness = np.max(self.hsml) + self.max_search_radius
        if pa.settings.use_units:
            self.extra_layer_thickness_value = self.extra_layer_thickness.value
        else:
            self.extra_layer_thickness_value = self.extra_layer_thickness

        # Create tiling
        if (tilingType == 'cartesian'):
            self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'], self.gpu_variables['widths'], self.extra_layer_thickness_value, npix=npix, threadsperblock=threadsperblock)


        # # Do the sorting
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
            self.indicesFirstPass = get_index(self.snap["0_Coordinates"],center, widths, thickness, snap.box)
            self.max_search_radius = np.max(self.search_radius[self.indicesFirstPass])
            thickness = self.hsml + self.max_search_radius
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness,
                                   snap.box)
        else:
            get_index = pa.util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.indicesFirstPass = get_index(self.snap["0_Coordinates"], center, widths, thickness, snap.box, self.orientation)
            self.max_search_radius = np.max(self.search_radius[self.indicesFirstPass])
            thickness = self.hsml + self.max_search_radius
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness, snap.box,
                                   self.orientation)
        nvtx.end_range(rng)

        self.pos = self.pos[self.index]
        self.hsml = self.hsml[self.index]

        self._send_data_to_gpu()


    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if pa.settings.use_units:
            self.gpu_variables['pos'] = cp.array(self.pos.value)
            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
            self.gpu_variables['widths'] = cp.array(self.widths.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            self.gpu_variables['pos'] = cp.array(self.pos)
            self.gpu_variables['hsml'] = cp.array(self.hsml)
            self.gpu_variables['widths'] = cp.array(self.widths)
            self.gpu_variables['center'] = cp.array(self.center)

        if self.orientation is not None:
            self.gpu_variables['rotation_matrix'] = cp.array(
                self.orientation.rotation_matrix)
            self.gpu_variables['inverse_rotation_matrix'] = cp.array(
                self.orientation.inverse_rotation_matrix)
            # rotate coordinates
            self.gpu_variables['pos'] = cp.matmul(self.gpu_variables['inverse_rotation_matrix'], self.gpu_variables['pos'], axes=[(-2, -1), (-1, -2), (-1, -2)])
            self.gpu_variables['center'] = cp.matmul(self.gpu_variables['inverse_rotation_matrix'], self.gpu_variables['center'])
            

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
                       selection=None):
        """
        variable can also be a vector
        """

            
        rng0 = nvtx.start_range(message="do_filter")

        if isinstance(variable, str):
            variable = self.snap[variable]

        # variable can also be a vector
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            self._send_variable_to_gpu(weight, gpu_key='weight')

        if not hasattr(filter_length, 'unit'):
            raise RuntimeError('filter_length must have unit')

        # send filter_length to gpu
        if isinstance(filter_length.value, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length.value

        if selection is not None:
            self._send_variable_to_gpu(selection*self.snap.uq(''), gpu_key='selection')

        # Do the filtering
        # do the optimized by default
        smooth_variable = self._apply_filter_gpu_optimized(variable_str, weight, filter_type, iterative)

        if unit_quantity is not None:
            smooth_variable = smooth_variable * unit_quantity

        nvtx.end_range(rng0)
        
        return smooth_variable

    def _apply_filter_gpu_optimized(self, variable_str, weight, filter_type, iterative):
        """
        The idea behind this 'optimized' version is to check if the particle
        is in the domain _beforehand_, and then run the filtering kernel 
        only on those that are in the domain. There is a certain speedup 
        in doing so (for small-size problems running time can be 1/3)
        I have also improved the tile searching within the kernel: now only
        the tiles that *overlap* with the filtering radius of each particle
        are selected, without wasting time looping over those that do not
        """
    
            
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
            weights = self.gpu_variables['weight']
        else:
            weights = cp.ones(pos.shape[0])

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
        hitsNeighbours = cp.zeros(pos.shape[0],dtype="int")
        # isParticleInDomain = cp.zeros(variable.shape,dtype="int")
        hasConverged = cp.zeros(pos.shape[0],dtype="int")
        numIterations = cp.zeros(pos.shape[0],dtype="int")
        filter_lengths_out = cp.zeros(pos.shape[0],dtype="float")

        rng = nvtx.start_range(message="cartesian filter (optimized)")
        # check if it is a vector:
        if (len(variable.shape) > 1):
            # is a vector
            # TODO: change definition of apply_filter 
            apply_filter_optimized_vector[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths, variable[:,0], variable[:,1], variable[:,2], weights, offsets, npixs, center, widths, filter_lengths, smooth_var[:,0], smooth_var[:,1], smooth_var[:,2], filter_type, hitsNeighbours,isParticleInDomain, iterativeFilter, hasConverged, numIterations, filter_lengths_out, self.multiplier, max_search_radius)
            
        else:
            # is a scalar
            apply_filter_optimized[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,variable, weights, offsets, npixs, center, widths, filter_lengths, smooth_var, filter_type, hitsNeighbours, isParticleInDomain, iterativeFilter, hasConverged,numIterations, filter_lengths_out, self.multiplier,max_search_radius)
        
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
            print("Min/Max iterations needed: %d / %d"%(np.min(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0]), np.max(self.numIterationsUnSorted[self.isParticleInDomainUnSorted>0])))
        
        
        return cp.asnumpy(smooth_var[self.tile.unsort_index])

    

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
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
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

    def _apply_derivative_gpu(self, variable_str, filter_type):
        """
        
        """

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
        
        new_shape = [i for i in variable.shape]
        new_shape.append(3)
        grad_var = cp.zeros(new_shape)
        
        hitsNeighbours = cp.zeros(variable.shape[0],dtype="int")
        hasConverged = cp.zeros(variable.shape[0],dtype="int")

        rng = nvtx.start_range(message="cartesian filter (optimized)")

        # check if it is a vector:
        if (len(variable.shape) > 1):
            # is a vector
            apply_derivative_vector[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths, variable[:,0], variable[:,1], variable[:,2], offsets, npixs, center, widths, filter_lengths, grad_var[:,0,0], grad_var[:,1,0], grad_var[:,2,0], grad_var[:,0,1], grad_var[:,1,1], grad_var[:,2,1], grad_var[:,0,2], grad_var[:,1,2], grad_var[:,2,2], filter_type, hitsNeighbours, isParticleInDomain, hasConverged, self.multiplier)
        else:
            # is a scalar
            apply_derivative[blocks_1d, self.threadsperblock](oldIndex, pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,variable, offsets, npixs, center, widths, filter_lengths, grad_var[:,0], grad_var[:,1], grad_var[:,2], filter_type, hitsNeighbours, isParticleInDomain, hasConverged, self.multiplier)
        
        nvtx.end_range(rng)

        self.hitsNeighbours = hitsNeighbours
        self.hitsNeighboursUnSorted = hitsNeighbours[self.tile.unsort_index]
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
        self.hasConvergedUnSorted = hasConverged[self.tile.unsort_index]
        

        grad_var = cp.asnumpy(grad_var[self.tile.unsort_index])
        
        return grad_var 

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
