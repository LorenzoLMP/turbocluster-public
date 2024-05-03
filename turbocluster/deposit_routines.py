import numpy as np
import cupy as cp
from numba import cuda
import nvtx
import paicos as pa

class DepositCartesianGrid:
    """
    """
    def __init__(self, snap, center, widths, orientation=None,
                 npix=128, threadsperblock=256, regionType='cartesian', rMin=-1.0, 
                 rMax=-1.0):

        if orientation is not None:
            raise RuntimeError('not implemented')

        if (regionType == 'spherical'):
            self.spherical = True
            self.cartesian = False
        elif (regionType == 'cartesian'):
            self.cartesian = True
            self.spherical = False

        if (regionType == 'spherical'):
            if (rMin < 0.0) or (rMax < 0.0) or (rMax < rMin):
                raise RuntimeError('With spherical \
                you need to provide a non-negative \
                rMin and rMax > rMin')

        
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

        if (regionType == 'spherical'):
            
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
        self.hsml = 2.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))

        if pa.settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)


        if (regionType == 'cartesian'):
            self._do_region_selection()
        elif (regionType == 'spherical'):
            self._do_region_selection_spherical()

        self.extra_layer_thickness = np.max(self.hsml) 
        if pa.settings.use_units:
            self.extra_layer_thickness_value = self.extra_layer_thickness.value
        else:
            self.extra_layer_thickness_value = self.extra_layer_thickness

        # Define uniform grid
        if (regionType == 'cartesian'):
            # if region is a parallelepiped with widths:
            self.tilebox_widths = self.gpu_variables['widths']
    
            npix_x = npix
            npix_y = int(self.tilebox_widths[1] / self.tilebox_widths[0] * npix)
            npix_z = int(self.tilebox_widths[2] / self.tilebox_widths[0] * npix)


        elif (regionType == 'spherical'):
            # if region is a spherical region with Rmax:
            self.tilebox_widths = cp.array([2.0 * rMax, 2.0 * rMax, 2.0 * rMax]) 
    
            npix_x = npix
            npix_y = npix
            npix_z = npix

        self.npixs = cp.array([npix_x, npix_y, npix_z])
        self.off_sets = self.gpu_variables['center'] - self.tilebox_widths / 2.0
        self.tile_widths = self.tilebox_widths / self.npixs

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock


    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        thickness = self.hsml 
        # rng = nvtx.start_range(message="region_selection")
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
        # nvtx.end_range(rng)

        self.pos = self.pos[self.index]
        self.hsml = self.hsml[self.index]

        self._send_data_to_gpu()

    def _do_region_selection_spherical(self):
        """ 
        
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