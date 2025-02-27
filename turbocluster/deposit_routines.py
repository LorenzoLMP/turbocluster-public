import numpy as np
import cupy as cp
from numba import cuda
import nvtx
import paicos as pa

from .deposit_kernels import *
from .generic_kernels import *
from .power_spectra_kernels import *

class DepositCartesianGrid:
    """
    """
    def __init__(self, snap, center, widths, orientation=None,
                 npoints=128, threadsperblock=256, regionType='cartesian', rMin=-1.0, 
                 rMax=-1.0, kernel_type='PCS'):

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

        # Calculate the diameter of the particle
        # self.hsml = 2.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))
        # test with twice as large diameter
        self.hsml = 4.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))

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


        if (regionType == 'cartesian'):
            self._do_region_selection()
        elif (regionType == 'spherical'):
            self._do_region_selection_spherical()

        self.extra_layer_thickness = np.max(self.hsml) 
        if pa.settings.use_units:
            self.extra_layer_thickness_value = self.extra_layer_thickness.value
        else:
            self.extra_layer_thickness_value = self.extra_layer_thickness

        
        npix = npoints - 1
        
        # Define uniform grid
        if (regionType == 'cartesian'):
            # if region is a parallelepiped with widths:
            self.tilebox_widths = self.gpu_variables['widths']

            # the tiles are npoints - 1
            
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
        self.npoints = cp.array([npoints, npix_y+1, npix_z+1])
        
        self.off_sets = self.gpu_variables['center'] - self.tilebox_widths / 2.0
        self.tile_widths = self.tilebox_widths / self.npixs

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock


    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap
        support = self.support
        # Send subset of snapshot to GPU
        # get the index of the region of projection
        # try with 5 to take into account that
        # support of particles can be larger than hsml
        # depending on the shape of the interpolant
        thickness = support*self.hsml 
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
        support = self.support
        
        # Send subset of snapshot to GPU
        # get the index of the region of projection
        thickness = support*self.hsml         
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
            self.gpu_variables[variable_str] = self.gpu_variables[variable_str]

        if isinstance(variable, pa.units.PaicosQuantity):
            unit_quantity = variable.unit_quantity
        else:
            unit_quantity = None

        return variable_str, unit_quantity

    def _do_deposition_gpu(self, variable_str, weight):
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        # tile_index = self.tile.tile_index
        # start_index_for_tile = self.tile.start_index_for_tile
        # particles_per_tile = self.tile.particles_per_tile
        variable = self.gpu_variables[variable_str]
        center = self.gpu_variables['center']
        offsets = self.off_sets
        
        if self.cartesian:
            tile_widths = self.tile_widths
            widths = self.gpu_variables['widths']
            npixs = self.npixs
        elif self.spherical:
            raise RuntimeError('Deposition with spherical \
                region not yet implemented')
            # _rMin = self.tile._rMin
            # _rMax = self.tile._rMax
            # rMin = self.rMin.value
            # rMax = self.rMax.value
            # nSects = self.tile.nSects
            # spacings = self.tile.spacings
            # typeGrid = self.typeGrid
            # power    = self.powerGrid
        
        kernel_type = self.support
        # filter_lengths = self.gpu_variables['filter_lengths']
        
            
        # deposited_var = cp.zeros((int(npixs[0]),int(npixs[1]),int(npixs[2])),
        #                          dtype="float")
        # scratch = cp.zeros((int(npixs[0]),int(npixs[1]),int(npixs[2])),
        #                    dtype="float")

        deposited_var = cp.zeros((int(self.npoints[0]),int(self.npoints[1]),int(self.npoints[2])),
                                 dtype="float")
        scratch = cp.zeros((int(self.npoints[0]),int(self.npoints[1]),int(self.npoints[2])),
                           dtype="float")

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)


        rng = nvtx.start_range(message="cartesian deposition")
        deposit_on_grid[self.blocks_1d, self.threadsperblock](pos, hsml, tile_widths,
                         variable, weights, offsets, npixs, center, widths, deposited_var, 
                         scratch, kernel_type)
        nvtx.end_range(rng)

        self.scratch = scratch
        self.deposited_var = deposited_var
        
        # return cp.asnumpy(deposited_var/scratch)
        if (np.argwhere(deposited_var==0.0).size > 0):
            print("Warning: %d grid points have zero values"%(np.argwhere(deposited_var==0.0).size))
        
        # return cp.asnumpy(cp.where(scratch>0,deposited_var/scratch,0.0))
        return cp.asnumpy(deposited_var)

    def deposit_variable(self, variable, weight=None):
        """
        
        """

            
        rng0 = nvtx.start_range(message="do_deposition")
        
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            if isinstance(weight, str):
                self._send_variable_to_gpu(weight)
            else:
                raise RuntimeError('has to be a string')

        deposited_variable = self._do_deposition_gpu(variable_str, weight)

        if unit_quantity is not None:
            deposited_variable = deposited_variable * unit_quantity

        nvtx.end_range(rng0)
        
        return deposited_variable

    def power_spectrum1d(self, deposited_variable, **kwargs):
        """
        kwargs:
            window : 1D window generation function
               The window function should accept one argument: the window length.
               Example: window=scipy.signal.windows.hann
        """
        
        Nx, Ny, Nz = deposited_variable.shape

        if pa.settings.use_units:
        # if hasattr(self.widths, 'unit'):
            Lx, Ly, Lz = self.widths.value
            L_unit = self.widths.unit
        else:
            Lx, Ly, Lz = self.widths

        if pa.settings.use_units:
            depo_variable = deposited_variable.value.copy()
            variable_unit = deposited_variable.unit
        else:
            depo_variable = deposited_variable.copy()

        voxel_real_space = (Lx/Nx)*(Ly/Ny)*(Lz/Nz)        
        energy_real_space = np.sum(depo_variable**2*voxel_real_space)
        print('energy (real space) = %.4e'%(energy_real_space))

        # this is for consistency with zero-padding
        if 'npads' in kwargs:
            Nx *= int(kwargs['npads'])
            Ny *= int(kwargs['npads'])
            Nz *= int(kwargs['npads'])
            Lx *= int(kwargs['npads'])
            Ly *= int(kwargs['npads'])
            Lz *= int(kwargs['npads'])

        # this is if we want to do windowing
        if 'window' in kwargs:
            depo_variable, ndim_window = nd_window(depo_variable, 
                                             kwargs["window"])
            
        # Send variable to gpu
        d_depo_variable = cp.array(depo_variable)
            
        hat_depo_variable = cp.fft.rfftn(d_depo_variable, s=(Nx,Ny,Nz))
        Ntotalcomplex = Nx*Ny*Nz
        
        ## create the wavevectors
        kx = 2.0*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
        ky = 2.0*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
        kz = 2.0*np.pi*np.fft.rfftfreq(Nz, d=Lz/Nz)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            
        K2 = KX**2 + KY**2 + KZ**2

        kvec = np.sqrt(KX**2 + KY**2 + KZ**2)
    
        kxmax = (2.0*np.pi/Lx)*(Nx//2)
        kymax = (2.0*np.pi/Ly)*(Ny//2)
        kzmax = (2.0*np.pi/Lz)*(Nz//2)
    
        kmax = np.sqrt(kxmax**2 + kymax**2 + kzmax**2)
        # I take the coarsest grid in k-space
        deltak = 2.0*np.pi/np.min([Lx,Ly,Lz])
        # this is to take into account that with zero-padding
        # Lx,Ly,Lz are greater but we still want to keep the
        # coarser base-grid for the power spectrum
        if 'npads' in kwargs:
            deltak *= int(kwargs['npads'])

        self.deltak = deltak

        nbin = int(kmax/deltak + 0.5)
        n1d = np.arange(0, nbin)
        k1d = deltak*n1d
    
        ## these are now the *wavenumbers*
        ## (not wavevectors) 
        wavenum = kvec / deltak
        ## store them on device
        d_wavenum = cp.array(wavenum)  
        
        d_powerspectr = cp.zeros(k1d.shape)
        blocks_1d = (Ntotalcomplex + (self.threadsperblock - 1)) // self.threadsperblock

        ## launch kernel
        gpu_power_spectrum1d[blocks_1d, self.threadsperblock](hat_depo_variable, d_wavenum, 
                                                              (Nx,Ny,Nz), 
                                                              d_powerspectr)

        powerspectr = cp.asnumpy(d_powerspectr)*(Lx*Ly*Lz)*(2.0*np.pi/deltak) ## for energy/frequency
        # powerspectr = cp.asnumpy(d_powerspectr)*(2.0*np.pi/deltak)    ## for energy/volume/frequency
        
        if 'window' in kwargs: ## for normalization with window function
            powerspectr /= np.sum(np.square(ndim_window))/(Nx*Ny*Nz)
        # if 'npads' in kwargs: # this is for consistency with zero-padding
        #     powerspectr *= int(kwargs['npads'])**3
    
        energy_fourier_space = np.sum(powerspectr*deltak/(2.0*np.pi))
        print('energy (fourier space) = %.4e'%(energy_fourier_space))
        
        if pa.settings.use_units:
            k1d /= L_unit
            powerspectr *= L_unit**4 # (for energy/frequency: 3 powers for Lx*Ly*Lz + 1 power for 1/deltak)
            # powerspectr *= L_unit # (for energy/volume/frequency: 1 power for 1/deltak)
            powerspectr *= variable_unit**2
        
        
        return powerspectr, k1d, (KX, KY, KZ, cp.asnumpy(hat_depo_variable))

        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("bye")
        del self.deposited_var
        del self.scratch
        del self.gpu_variables
        del self.index
        del self.pos
        del self.hsml
        del self.snap
        del self.support
        del self.off_sets
        del self.center
        del self.widths