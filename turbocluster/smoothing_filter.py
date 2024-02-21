import numpy as np
import cupy as cp
from numba import cuda
import numba
from .. import util
from .. import settings
from .. import units


class SmoothingFilter:
    """
    """

    def __init__(self, snap, center, widths, orientation=None,
                 max_filter_length=None, npix=128, threadsperblock=8):

        if orientation is not None:
            raise RuntimeError('not implemented')
        if max_filter_length is None:
            raise RuntimeError('need input')

        self.max_filter_length = max_filter_length

    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        if self.orientation is None:
            get_index = util.get_index_of_cubic_region_plus_thin_layer
            self.index = get_index(self.snap["0_Coordinates"],
                                   center, widths, self.max_filter_length,
                                   snap.box)
        else:
            get_index = util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.index = get_index(snap["0_Coordinates"],
                                   center, widths, self.max_filter_length, snap.box,
                                   self.orientation)

        self.pos = self.pos[self.index]

        self._send_data_to_gpu()

    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if settings.use_units:
            self.gpu_variables['pos'] = cp.array(self.pos.value)
        else:
            self.gpu_variables['pos'] = cp.array(self.pos)

        if self.orientation is not None:
            self.gpu_variables['rotation_matrix'] = cp.array(
                self.orientation.rotation_matrix)

        if settings.use_units:
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
