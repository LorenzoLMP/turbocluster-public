import numpy as np
import cupy as cp
# from numba import cuda
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling


class SmoothingFilter:
    """
    """

    def __init__(self, snap, center, widths, orientation=None,
                 max_filter_length=None, npix=128, threadsperblock=256):

        if orientation is not None:
            raise RuntimeError('not implemented')
        if max_filter_length is None:
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

        if hasattr(max_filter_length, 'unit'):
            self.max_filter_length = max_filter_length.copy
            assert max_filter_length.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            self.max_filter_length = np.array(max_filter_length) * code_length
        else:
            self.max_filter_length = np.array(max_filter_length)

        if orientation is None:
            self.orientation = None
        else:
            self.orientation = orientation.copy

        self.pos = self.snap["0_Coordinates"]

        self._do_region_selection()

        # Create tiling
        self.tile = CartesianTiling(self.pos, npix=npix,
                                    threadsperblock=threadsperblock)

        self.gpu_variables['pos'] = self.gpu_variables['pos'][self.tile.sort_index, :]

        Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock

    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        ones = np.ones(self.snap["0_Coordinates"].shape[0])
        thickness = self.max_filter_length * ones
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

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        del self.gpu_variables
        cp._default_memory_pool.free_all_blocks()
