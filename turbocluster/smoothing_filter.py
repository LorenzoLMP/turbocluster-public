import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling

# these two raise an error
# from smoothing_filter_shared import compute_max_hsml, check_block, compactify_in_domain, apply_filter_shared
# import smoothing_filter_shared


@cuda.jit()
def compute_max_hsml(hsml,compactGrid,maxHsmlPerBlock):
    """
    """
    ip = cuda.grid(1)
    # each thread is assigned to an occupied block
    numBlocksCompGrid = compactGrid.shape[0]
    hsmlMaxTmp = 0.0
    
    if (ip < numBlocksCompGrid):
        # loop over the particles contained in that block
        for i in range(compactGrid[ip, 2]):
            # compactGrid[ip, 1] contains the first particle Id 
            particleIp = compactGrid[ip, 1] + i
            # hsmlMaxTmp = (hsml[particleIp] > hsmlMaxTmp) ? hsml[particleIp] : hsmlMaxTmp
            hsmlMaxTmp = hsml[particleIp] if (hsml[particleIp] > hsmlMaxTmp) else hsmlMaxTmp

        maxHsmlPerBlock[ip] = hsmlMaxTmp
        
@cuda.jit()
def check_block(pos, hsml, compactGrid, center, widths, isBlockInDomain):
    """
    """
    ip = cuda.grid(1)
    # each thread is assigned to an occupied block
    numBlocksCompGrid = compactGrid.shape[0]
    isBlockInDomainTmp = 0
    
    if (ip < numBlocksCompGrid):
        # loop over the particles contained in that block
        for i in range(compactGrid[ip, 2]):
            # compactGrid[ip, 1] contains the first particle Id 
            particleIp = compactGrid[ip, 1] + i
            xp, yp, zp = pos[particleIp]
            xmin = center[0] - widths[0] / 2 - 2.0 * hsml[particleIp]
            xmax = center[0] + widths[0] / 2 + 2.0 * hsml[particleIp]
        
            ymin = center[1] - widths[1] / 2 - 2.0 * hsml[particleIp]
            ymax = center[1] + widths[1] / 2 + 2.0 * hsml[particleIp]
        
            zmin = center[2] - widths[2] / 2 - 2.0 * hsml[particleIp]
            zmax = center[2] + widths[2] / 2 + 2.0 * hsml[particleIp]

            if (xp > xmin) and (xp < xmax):
                if (yp > ymin) and (yp < ymax):
                    if (zp > zmin) and (zp < zmax):
                        isBlockInDomainTmp = 1
            

        isBlockInDomain[ip] = isBlockInDomainTmp
        
@cuda.jit()
def compactify_in_domain(fullCompactGrid, cumulative_occupancy, isBlockInDomain, compactGridInDomain):
    """
    """
    ip = cuda.grid(1)
    numBlocksFullCompactGrid = fullCompactGrid.shape[0]
    if (ip < numBlocksFullCompactGrid):
        if (isBlockInDomain[ip]):
            newPos = int(cumulative_occupancy[ip]) - 1
            compactGridInDomain[newPos, 0] = fullCompactGrid[ip, 0]
            compactGridInDomain[newPos, 1] = fullCompactGrid[ip, 1]
            compactGridInDomain[newPos, 2] = fullCompactGrid[ip, 2]         
        


@cuda.jit()
def apply_filter_shared(compactGrid, pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,
                 variable, weights, center, widths, npixs, filter_lengths, smooth_var, filter_type):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    """
    # threadindex (absolute position within all blocks) equal to cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ipAbs = cuda.grid(1)
    # relative position of thread in a block
    threadId = cuda.threadIdx.x
    # block id
    blockId = cuda.blockIdx.x
    # num threads in the Block
    numThreads = cuda.blockDim.x

    center_x, center_y, center_z = center
    widths_x, widths_y, widths_z = widths
    tile_widths_x, tile_widths_y, tile_widths_z = tile_widths
    nx, ny, nz = npixs

    # id (3D) of block: which is the original tile of the
    # cartesian grid it refers to. 
    # blockID3D = blockID_z + nz * (blockID_y + ny * blockID_x)
    blockID3D = compactGrid[blockId, 0]
    # starting index of the particle in this block
    startIdxBlock = compactGrid[blockId, 1]
    # how many particles are there in the block
    numParticlesOwnBlock = compactGrid[blockId, 2]
    
    # blockID_z = blockID3D % int(nz)
    # new_id = int((blockID3D - blockID_z)/int(nz))
    # blockID_y = new_id % int(ny)
    # blockID_x = new_id // int(ny)

    ipTile_x, ipTile_y, ipTile_z = tile_index[startIdxBlock]

    # find all the particles from neighbouring tiles that may overlap
    # with those in this block

    # I need a maxFilterLength per block in order to establish
    # all overlapping neighbouring tiles
    maxFilterLength = cuda.shared.array(1, dtype="float64")
    if threadId == 0:
        maxFilterLength[0] = 0.0
    cuda.syncthreads()
    # do some kind of atomic add using all available threads
    if (threadId < numParticlesOwnBlock):
        idParticle = startIdxBlock + threadId
        cuda.atomic.max(maxFilterLength, 0, filter_lengths[idParticle])

    cuda.syncthreads()
    maxFilterLength = maxFilterLength.item()

    filter_window = 1.0
    # for gaussian filter we actually want to look for particles up to 4 times
    # filter_length far away from the source particle
    if filter_type == 1:
        filter_window = 4.0

    ip_tile_x_min = ipTile_x - \
        int((filter_window * maxFilterLength) / tile_widths_x + 1)
    ip_tile_x_max = ipTile_x + \
        int((filter_window * maxFilterLength) / tile_widths_x + 1)

    ip_tile_y_min = ipTile_y - \
        int((filter_window * maxFilterLength) / tile_widths_y + 1)
    ip_tile_y_max = ipTile_y + \
        int((filter_window * maxFilterLength) / tile_widths_y + 1)

    ip_tile_z_min = ipTile_z - \
        int((filter_window * maxFilterLength) / tile_widths_z + 1)
    ip_tile_z_max = ipTile_z + \
        int((filter_window * maxFilterLength) / tile_widths_z + 1)

    numTotalNeighbours = int((ip_tile_x_max - ip_tile_x_min) * \
                         (ip_tile_y_max - ip_tile_y_min) * \
                         (ip_tile_z_max - ip_tile_z_min))
    



    # https://numba.readthedocs.io/en/stable/cuda/memory.html#dynamic-shared-memory
    # https://curiouscoding.nl/posts/numba-cuda-speedup/#v15-dynamic-shared-memory
    # Note that all dynamic shared memory arrays alias, so 
    # if you want to have multiple dynamic shared arrays, you need 
    # to take disjoint views of the arrays
    # these dynamically allocated arrays are now one-dimensional
    # so we need to index them accordingly
    dynamicAllBuffer = cuda.shared.array(shape=0, dtype="float64")
    ownParticle = dynamicAllBuffer[0:int(numThreads*7)]
    neighParticleBuf = dynamicAllBuffer[int(numThreads*7):]

    # copy own particles into shared memory
    if (threadId < numParticlesOwnBlock):
        idParticle = startIdxBlock + threadId
        pos_x, pos_y, pos_z =  pos[idParticle]
        ownParticle[threadId*7 + 0] = pos_x
        ownParticle[threadId*7 + 1] = pos_y
        ownParticle[threadId*7 + 2] = pos_z
        ownParticle[threadId*7 + 3] = weights[idParticle]
        ownParticle[threadId*7 + 4] = variable[idParticle]
        ownParticle[threadId*7 + 5] = filter_lengths[idParticle]
        ownParticle[threadId*7 + 6] = hsml[idParticle]
        

    cuda.syncthreads()

    # start the actual filtering: at every iteration load at most
    # N <= numThreads particles of a neighbouring tile
    # from global memory to shared memory in the 
    # neighbouring Particle Buffer
    # then for each particle in the block (assigned to a thread), iterate
    # over N to compute the contributions between particles in own block and
    # those neighbouring just copied

    weight = 0.0
    smoothVarRegister = 0.0
    for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
        for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
            for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                numNeighParticles = particles_per_tile[tile_x, tile_y,
                                        tile_z]
                startIdNeigh = start_index_for_tile[tile_x, tile_y,
                                        tile_z]
                remainingParticles = numNeighParticles
                numIterations = (numNeighParticles + (numThreads - 1)) // numThreads
                for iter in range(numIterations):
                    # copy from neighbour tile
                    if (threadId < remainingParticles):
                        idParticle = startIdNeigh + threadId + iter * numThreads
                        pos_x, pos_y, pos_z =  pos[idParticle]
                        neighParticleBuf[threadId*5 + 0] = pos_x
                        neighParticleBuf[threadId*5 + 1] = pos_y
                        neighParticleBuf[threadId*5 + 2] = pos_z
                        neighParticleBuf[threadId*5 + 3] = weights[idParticle]
                        neighParticleBuf[threadId*5 + 4] = variable[idParticle]
            
                    cuda.syncthreads()
                    numParticlesLoaded = numThreads if (remainingParticles >= numThreads) \
                                        else remainingParticles
                    remainingParticles -= numThreads

                    # compute kernel overlap between loaded particles
                    # and own particles
                    if (threadId < numParticlesOwnBlock):    
                        idOwnParticle = startIdxBlock + threadId
                        ownFilterLength = ownParticle[threadId*7 + 5]
                        for ip in range(numParticlesLoaded):
                            ipShifted = (ip + threadId) % numParticlesLoaded
                            dist = distance(ownParticle[threadId*7:threadId*7 + 3], 
                                            neighParticleBuf[ip*5:ip*5 + 3])
                            # dist = distance(ownParticle[threadId*7:threadId*7 + 3], 
                            #                 neighParticleBuf[ipShifted*5:ipShifted*5 + 3])
                            if dist < ownFilterLength:
                                weight_tmp = 1.0 * neighParticleBuf[ip*5 + 3]
                                # weight_tmp = 1.0 * neighParticleBuf[ipShifted*5 + 3]
                                weight += weight_tmp
                                smoothVarRegister += neighParticleBuf[ip*5 + 4] * weight_tmp
                                # smoothVarRegister += neighParticleBuf[ipShifted*5 + 4] * weight_tmp
            
                    cuda.syncthreads()


    if (threadId < numParticlesOwnBlock):   
        # weight should never be zero since any particle finds at
        # least itself
        if weight > 0:
            smoothVarRegister /= weight
        
        # now we need to write smoothVarRegister back into 
        # global memory. It could be that this particle (thread)
        # is actually outside the filtering domain. So we need to 
        # check first
        idOwnParticle = startIdxBlock + threadId

        xmin = center_x - widths_x / 2 - 2.0 * ownParticle[threadId*7 + 6]
        xmax = center_x + widths_x / 2 + 2.0 * ownParticle[threadId*7 + 6]
    
        ymin = center_y - widths_y / 2 - 2.0 * ownParticle[threadId*7 + 6]
        ymax = center_y + widths_y / 2 + 2.0 * ownParticle[threadId*7 + 6]
    
        zmin = center_z - widths_z / 2 - 2.0 * ownParticle[threadId*7 + 6]
        zmax = center_z + widths_z / 2 + 2.0 * ownParticle[threadId*7 + 6]

        xp, yp, zp = ownParticle[threadId*7:threadId*7 + 3]

        if ((xp > xmin) and (xp < xmax) 
            and (yp > ymin) and (yp < ymax) 
                and (zp > zmin) and (zp < zmax)):
                    smooth_var[idOwnParticle] = smoothVarRegister



@cuda.jit()
def apply_filter(pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,
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

    xmin = center[0] - widths[0] / 2 - 2.0 * hsml[ip]
    xmax = center[0] + widths[0] / 2 + 2.0 * hsml[ip]

    ymin = center[1] - widths[1] / 2 - 2.0 * hsml[ip]
    ymax = center[1] + widths[1] / 2 + 2.0 * hsml[ip]

    zmin = center[2] - widths[2] / 2 - 2.0 * hsml[ip]
    zmax = center[2] + widths[2] / 2 + 2.0 * hsml[ip]

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
                                weight_tmp = 1.0 * weights[ip_other]
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
                                weight_tmp = gaussian_kernel(dist, filter_length) * weights[ip_other]
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

    def __init__(self, snap, center, widths, orientation=None, max_search_radius=None,
                 npix=128, threadsperblock=256):

        if orientation is not None:
            raise RuntimeError('not implemented')
        # if max_search_radius is None:
        #     raise RuntimeError('need input')

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

        if max_search_radius is None:
            max_search_radius = 0.2 * np.max(self.widths)

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

        # Calculate the smoothing length
        self.hsml = 2.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))

        if pa.settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        self._do_region_selection()

        self.extra_layer_thickness = np.max(2 * self.hsml) + self.max_search_radius
        if pa.settings.use_units:
            self.extra_layer_thickness_value = self.extra_layer_thickness.value
        else:
            self.extra_layer_thickness_value = self.extra_layer_thickness

        # Create tiling
        self.tile = CartesianTiling(self.gpu_variables['pos'], self.gpu_variables['center'],
                                    self.gpu_variables['widths'], self.extra_layer_thickness_value, npix=npix,
                                    threadsperblock=threadsperblock)

        # Do the sorting
        self.gpu_variables['pos'] = self.gpu_variables['pos'][self.tile.sort_index, :]
        self.gpu_variables['hsml'] = self.gpu_variables['hsml'][self.tile.sort_index]

        self.Np = Np = self.gpu_variables['pos'].shape[0]

        self.blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock
        self.threadsperblock = threadsperblock

    def _do_region_selection(self):

        center = self.center
        widths = self.widths
        snap = self.snap

        # Send subset of snapshot to GPU
        # get the index of the region of projection
        thickness = 2 * self.hsml + self.max_search_radius
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
            self.gpu_variables['widths'] = cp.array(self.widths.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            self.gpu_variables['widths'] = cp.array(self.widths)
            self.gpu_variables['center'] = cp.array(self.center)

    def _apply_filter_gpu(self, variable_str, weight, filter_type):
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
        filter_lengths = self.gpu_variables['filter_lengths']
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        smooth_var = cp.zeros_like(variable)

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        apply_filter[self.blocks_1d, self.threadsperblock](pos, hsml, tile_index, start_index_for_tile,
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

    def filter_variable(self, variable, filter_length, weight=None, filter_type="mean", iterative=False,
                       shared_mem=False, Nmax=64):
        """
        shared_mem has been tested only with filter_type="mean"
        Nmax is the max number of particles per block. Each tile is split
        in "logic" blocks with Nmax particles max (can be less, but not zero)
        and assigned to exactly 1 block of threads with Nmax threads
        """
        variable_str, unit_quantity = self._send_variable_to_gpu(variable)

        if weight is not None:
            if isinstance(weight, str):
                self._send_variable_to_gpu(weight)
            else:
                raise RuntimeError('has to be a string')

        if shared_mem and filter_type == "gaussian":
            raise RuntimeError('shared_mem has been tested only \
                                with filter_type="mean"')

        # send filter_length to gpu
        if isinstance(filter_length, np.ndarray):
            assert filter_length.shape[0] == self.index.shape[0]
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length

        # Do the filtering
        if not shared_mem:
            if not iterative:
                smooth_variable = self._apply_filter_gpu(variable_str, weight, filter_type)
            else:
                smooth_variable = self._apply_filter_gpu_iterative(variable_str, weight, filter_type)
        else:
            smooth_variable = self._apply_filter_gpu_shared(variable_str, weight, filter_type, Nmax)

        if unit_quantity is not None:
            smooth_variable = smooth_variable * unit_quantity

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
        if filter_type == "gaussian":
            raise RuntimeError('shared_mem has been tested only \
                                with filter_type="mean"')
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

        apply_filter_shared[numBlocksInDomain, Nmax, 0, sharedMemBuf](compactGrid, pos, hsml, tile_index, 
                                                     start_index_for_tile, particles_per_tile, 
                                                     tile_widths, variable, weights, center, 
                                                     widths, npixs, filter_lengths, smooth_var, filter_type)

        return cp.asnumpy(smooth_var[self.tile.unsort_index])

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        del self.gpu_variables
        # cp._default_memory_pool.free_all_blocks()
