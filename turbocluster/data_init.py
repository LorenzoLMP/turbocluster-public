import numpy as np
import cupy as cp
from numba import cuda
import math
import paicos as pa
import nvtx


from .cartesian_tiling import CartesianTiling


class DataGpuInit:
    """ """

    def __init__(
        self, snap, center, widths, orientation=None, threadsperblock=256, **kwargs
    ):
        """ """

        # self.__dict__.update(kwargs)

        self.snap = snap
        self.cartesian = True
        self.code_length = code_length = self.snap.length
        # self.npix = npix
        self.threadsperblock = threadsperblock

        if hasattr(center, "unit"):
            self.center = center.copy
            assert center.unit == code_length.unit, "this restriction applies"
        elif pa.settings.use_units:
            self.center = np.array(center) * code_length
        else:
            self.center = np.array(center)

        if hasattr(widths, "unit"):
            self.widths = widths.copy
            assert widths.unit == code_length.unit, "this restriction applies"
        elif pa.settings.use_units:
            self.widths = np.array(widths) * code_length
        else:
            self.widths = np.array(widths)

        if orientation is None:
            self.orientation = None
        else:
            self.orientation = orientation.copy

        # self.pos = self.snap["0_Coordinates"]

        self.gpu_variables = {}

    def _do_region_selection(self, thickness, pos):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        rng = nvtx.start_range(message="region_selection")
        if self.orientation is None:
            get_index = pa.util.get_index_of_cubic_region_plus_thin_layer
            indices = get_index(pos, center, widths, thickness, snap.box)
        else:
            get_index = pa.util.get_index_of_rotated_cubic_region_plus_thin_layer
            indices = get_index(
                pos, center, widths, thickness, snap.box, self.orientation
            )
        nvtx.end_range(rng)

        return indices

    def _send_variable_to_gpu(self, variable, gpu_key="input_variable", sort=False):
        if isinstance(variable, str):
            # variable_str = str(variable)
            # err_msg = 'filter only works on gas'
            # assert int(variable[0]) == 0, err_msg
            variable = self.snap[variable]
        else:
            # variable_str = gpu_key
            if not isinstance(variable, np.ndarray):
                raise RuntimeError("Unexpected type for variable")

        variable_str = gpu_key
        # select only region of interest
        if variable.shape[0] == self.index.shape[0]:
            variable = variable[self.index]

        if variable_str in self.gpu_variables and variable_str != gpu_key:
            pass
        else:
            # Send variable to gpu
            if pa.settings.use_units:
                self.gpu_variables[variable_str] = cp.array(variable.value)
            else:
                self.gpu_variables[variable_str] = cp.array(variable)

        if sort:
            try:
                if (
                    self.gpu_variables[variable_str].shape[0]
                    == self.tile.sort_index.shape[0]
                ):
                    self.gpu_variables[variable_str] = self.gpu_variables[variable_str][
                        self.tile.sort_index
                    ]
            except RuntimeError:
                raise RuntimeError(
                    "Sorted can be done only after creating a CartesianTiling"
                )

        if isinstance(variable, pa.units.PaicosQuantity):
            unit_quantity = variable.unit_quantity
        else:
            unit_quantity = None

        return variable_str, unit_quantity

    def _rotate_coordinates(self):

        if self.orientation is not None:
            if "inverse_rotation_matrix" not in self.gpu_variables:
                self.gpu_variables["inverse_rotation_matrix"] = cp.array(
                    self.orientation.inverse_rotation_matrix
                )

                self.gpu_variables["pos"] = cp.matmul(
                    self.gpu_variables["inverse_rotation_matrix"],
                    self.gpu_variables["pos"],
                    axes=[(-2, -1), (-1, -2), (-1, -2)],
                )

                self.gpu_variables["center"] = cp.matmul(
                    self.gpu_variables["inverse_rotation_matrix"],
                    self.gpu_variables["center"],
                )

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        self.release_gpu_memory()

    def release_gpu_memory(self):
        # TODO: Add deletion of all GPU variables stored in self
        if hasattr(self, "gpu_variables"):
            for key in list(self.gpu_variables):
                del self.gpu_variables[key]
            del self.gpu_variables
        if hasattr(self, "tile"):
            self.tile.release_gpu_memory()
            del self.tile

        # cp._default_memory_pool.free_all_blocks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("bye")
        self.__del__()
