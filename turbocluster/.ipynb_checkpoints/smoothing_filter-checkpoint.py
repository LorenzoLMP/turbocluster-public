import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling


@cuda.jit()
def apply_filter(pos, tile_index, start_index_for_tile, particles_per_tile, tile_widths,
                 variable, weights, npixs, center, widths, filter_lengths, smooth_var, filter_type):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    """
    # threadindex
    ip = cuda.grid(1)

    # particle position
    xp = pos[ip, 0]
    yp = pos[ip, 1]
    zp = pos[ip, 2]

    xmin = center[0] - widths[0] / 2
    xmax = center[0] + widths[0] / 2

    ymin = center[1] - widths[1] / 2
    ymax = center[1] + widths[1] / 2

    zmin = center[2] - widths[2] / 2
    zmax = center[2] + widths[2] / 2

    # in theory we can have different filter lengths per particle
    # for the iterative scheme in Vazza this number is gradually increased
    # maybe this function needs to be reworked in that case...
    filter_length = filter_lengths[ip]

    sidelength_x, sidelength_y, sidelength_z = widths
    nx, ny, nz = npixs

    # Check if this cell/particle is inside domain
    inside_domain = False
    if (xp > xmin) and (xp < xmax):
        if (yp > ymin) and (yp < ymax):
            if (zp > zmin) and (zp < zmax):
                inside_domain = True

    if inside_domain:

        ip_tile_x = tile_index[ip, 0]
        ip_tile_y = tile_index[ip, 1]
        ip_tile_z = tile_index[ip, 2]

        # tile_pos = tile_positions[tile_x, tile_y, tile_z]
        # tile_widths
        weight = 0.0
        weight_tmp = 0.0

        filter_window = 1
        # for gaussian filter we actually want to look for particles up to 4 times
        # filter_length far away from the source particle
        if filter_type == 1:
            filter_window = 4

        # this could be made smarter if one finds just the tiles that intersect
        # the sphere of radius filter_length
        # ip_tile_x_min = ((xp - xmin) - filter_window*filter_length)//tile_widths[0]
        # ip_tile_x_max = ((xp - xmin) + filter_window*filter_length)//tile_widths[0]

        # ip_tile_y_min = ((yp - ymin) - filter_window*filter_length)//tile_widths[1]
        # ip_tile_y_max = ((yp - ymin) + filter_window*filter_length)//tile_widths[1]

        # ip_tile_z_min = ((zp - zmin) - filter_window*filter_length)//tile_widths[2]
        # ip_tile_z_max = ((zp - zmin) + filter_window*filter_length)//tile_widths[2]

        ip_tile_x_min = ip_tile_x - \
            int((filter_window * filter_length) / tile_widths[0] + 1)
        ip_tile_x_max = ip_tile_x + \
            int((filter_window * filter_length) / tile_widths[0] + 1)

        ip_tile_y_min = ip_tile_y - \
            int((filter_window * filter_length) / tile_widths[1] + 1)
        ip_tile_y_max = ip_tile_y + \
            int((filter_window * filter_length) / tile_widths[1] + 1)

        ip_tile_z_min = ip_tile_z - \
            int((filter_window * filter_length) / tile_widths[2] + 1)
        ip_tile_z_max = ip_tile_z + \
            int((filter_window * filter_length) / tile_widths[2] + 1)

        if filter_type == 0:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        # tile_x = ip_tile_x
                        # tile_y = ip_tile_y
                        # tile_z = ip_tile_z
                        start_index = start_index_for_tile[tile_x,
                                                           tile_y, tile_z]
                        n_particles = particles_per_tile[tile_x,
                                                         tile_y, tile_z]

                        for ip_other in range(start_index, start_index + n_particles):
                            dist = distance(pos[ip], pos[ip_other])
                            if dist < filter_length:
                                weight_tmp = 1.0 * weights[ip]
                                weight += weight_tmp
                                smooth_var[ip] += variable[ip_other] * weight_tmp

        elif filter_type == 1:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        start_index = start_index_for_tile[tile_x,
                                                           tile_y, tile_z]
                        n_particles = particles_per_tile[tile_x,
                                                         tile_y, tile_z]

                        for ip_other in range(start_index, start_index + n_particles):
                            dist = distance(pos[ip], pos[ip_other])
                            if dist < filter_window * filter_length:
                                weight_tmp = gaussian_kernel(dist, filter_length) * weights[ip]
                                weight += weight_tmp
                                smooth_var[ip] += variable[ip_other] * weight_tmp
        if weight > 0.:
            smooth_var[ip] /= weight


@cuda.jit(device=True, inline=True)
def distance(pos, pos_other):
    dist = math.sqrt((pos[0] - pos_other[0])**2 +
                     (pos[1] - pos_other[1])**2 +
                     (pos[2] - pos_other[2])**2)
    return dist


@cuda.jit(device=True, inline=True)
def gaussian_kernel(dist, filter_length):

    weight = math.exp(-0.5*(dist/filter_length)**2)

    return weight


class SmoothingFilter:
    """
    """

    def __init__(self, snap, center, widths, orientation=None,
                 max_search_radius=None, npix=128, threadsperblock=256):

        if orientation is not None:
            raise RuntimeError('not implemented')
        if max_search_radius is None:
            raise RuntimeError('need input')

        self.snap = snap

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

        if hasattr(max_search_radius, 'unit'):
            self.max_search_radius = max_search_radius.copy
            assert max_search_radius.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            self.max_search_radius = np.array(max_search_radius) * code_length
        else:
            self.max_search_radius = np.array(max_search_radius)

        if orientation is None:
            self.orientation = None
        else:
            self.orientation = orientation.copy

        self.pos = self.snap["0_Coordinates"]

        self._do_region_selection()

        # Create tiling
        self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'],
                                    self.gpu_variables['widths'], max_search_radius, npix=npix,
                                    threadsperblock=threadsperblock)

        self.gpu_variables['pos'] = self.gpu_variables['pos'][self.tile.sort_index, :]

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock

    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        ones = np.ones(self.snap["0_Coordinates"].shape[0])
        thickness = self.max_search_radius * ones
        if self.orientation is None:
            get_index = pa.util.get_index_of_cubic_region_plus_thin_layer
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness,
                                   snap.box)
        else:
            get_index = pa.util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, thickness, snap.box,
                                   self.orientation)

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
            self.gpu_variables['widths'] = cp.array(self.widths.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            self.gpu_variables['widths'] = cp.array(self.widths)
            self.gpu_variables['center'] = cp.array(self.center)

    def _apply_filter_gpu(self, variable_str, weight, filter_type):
        pos = self.gpu_variables['pos']
        # - self.tile.off_sets[None,:]
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
        smooth_var = cp.zeros_like(variable)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        apply_filter[self.blocks_1d, self.threadsperblock](pos, tile_index, start_index_for_tile,
                                                           particles_per_tile, tile_widths,
                                                           variable, weights, npixs, center, widths, filter_lengths, smooth_var, filter_type)

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

    def filter_variable(self, variable, filter_length, weight=None, filter_type="mean", iterative=False):
        """

        """

        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            if isinstance(weight, str):
                self._send_variable_to_gpu(weight)
            else:
                raise RuntimeError('has to be a string')

        # send filter_length to gpu
        if isinstance(filter_length, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length

        # Do the filtering
        if not iterative:
            smooth_variable = self._apply_filter_gpu(variable_str, weight, filter_type)
        else:
            smooth_variable = self._apply_filter_gpu_iterative(variable_str, weight, filter_type)

        if unit_quantity is not None:
            smooth_variable = smooth_variable * unit_quantity

        return smooth_variable

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        del self.gpu_variables
        cp._default_memory_pool.free_all_blocks()
