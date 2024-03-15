import numpy as np
import cupy as cp
from numba import cuda
import math
# import numba
import paicos as pa
from .cartesian_tiling import CartesianTiling
from .spherical_tiling import SphericalTiling

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
            xmin = center[0] - widths[0] / 2 - hsml[particleIp]
            xmax = center[0] + widths[0] / 2 + hsml[particleIp]
        
            ymin = center[1] - widths[1] / 2 - hsml[particleIp]
            ymax = center[1] + widths[1] / 2 + hsml[particleIp]
        
            zmin = center[2] - widths[2] / 2 - hsml[particleIp]
            zmax = center[2] + widths[2] / 2 + hsml[particleIp]

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

    filter_window = 1
    # for gaussian filter the sigma is  1/4 of the
    # filter_length of the source particle
    if filter_type == 1:
        filter_window = 4

    ip_tile_x_min = ipTile_x - \
        int((maxFilterLength) / tile_widths_x + 1)
    ip_tile_x_max = ipTile_x + \
        int((maxFilterLength) / tile_widths_x + 1)

    ip_tile_y_min = ipTile_y - \
        int((maxFilterLength) / tile_widths_y + 1)
    ip_tile_y_max = ipTile_y + \
        int((maxFilterLength) / tile_widths_y + 1)

    ip_tile_z_min = ipTile_z - \
        int((maxFilterLength) / tile_widths_z + 1)
    ip_tile_z_max = ipTile_z + \
        int((maxFilterLength) / tile_widths_z + 1)

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

        xmin = center_x - widths_x / 2 - ownParticle[threadId*7 + 6]
        xmax = center_x + widths_x / 2 + ownParticle[threadId*7 + 6]
    
        ymin = center_y - widths_y / 2 - ownParticle[threadId*7 + 6]
        ymax = center_y + widths_y / 2 + ownParticle[threadId*7 + 6]
    
        zmin = center_z - widths_z / 2 - ownParticle[threadId*7 + 6]
        zmax = center_z + widths_z / 2 + ownParticle[threadId*7 + 6]

        xp, yp, zp = ownParticle[threadId*7:threadId*7 + 3]

        if ((xp > xmin) and (xp < xmax) 
            and (yp > ymin) and (yp < ymax) 
                and (zp > zmin) and (zp < zmax)):
                    smooth_var[idOwnParticle] = smoothVarRegister

@cuda.jit(device=True, inline=True)
def check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                   tile_x, tile_y, tile_z,
                   delta_x, delta_y, delta_z,
                   tile_widths, filter_length):

    overlap = False

    xcoord_edge = delta_x
    if (tile_x > ip_tile_x):
        xcoord_edge = tile_widths[0] * (tile_x - ip_tile_x - 0.5)
    elif (tile_x < ip_tile_x):
        xcoord_edge = tile_widths[0] * (tile_x - ip_tile_x + 0.5)

    ycoord_edge = delta_y
    if (tile_y > ip_tile_y):
        ycoord_edge = tile_widths[1] * (tile_y - ip_tile_y - 0.5)
    elif (tile_y < ip_tile_y):
        ycoord_edge = tile_widths[1] * (tile_y - ip_tile_y + 0.5)

    zcoord_edge = delta_z
    if (tile_z > ip_tile_z):
        zcoord_edge = tile_widths[2] * (tile_z - ip_tile_z - 0.5)
    elif (tile_z < ip_tile_z):
        zcoord_edge = tile_widths[2] * (tile_z - ip_tile_z + 0.5)
        
    dist2 = (delta_x - xcoord_edge)**2 + \
            (delta_y - ycoord_edge)**2 + \
            (delta_z - zcoord_edge)**2

    filt2 = filter_length**2
    if (filt2 >= dist2):
        overlap = True
    
    return overlap

@cuda.jit()
def check_particle(pos, hsml, center, widths, isParticleInDomain):
    """
    """
    ip = cuda.grid(1)
    # each thread is assigned to a particle
    numParticles = pos.shape[0]
    isParticleInDomainTmp = 0
    
    if (ip < numParticles):

        xp, yp, zp = pos[ip]
        xmin = center[0] - widths[0] / 2 - hsml[ip]
        xmax = center[0] + widths[0] / 2 + hsml[ip]
    
        ymin = center[1] - widths[1] / 2 - hsml[ip]
        ymax = center[1] + widths[1] / 2 + hsml[ip]
    
        zmin = center[2] - widths[2] / 2 - hsml[ip]
        zmax = center[2] + widths[2] / 2 + hsml[ip]

        if (xp > xmin) and (xp < xmax):
            if (yp > ymin) and (yp < ymax):
                if (zp > zmin) and (zp < zmax):
                    isParticleInDomainTmp = 1
            

        isParticleInDomain[ip] = isParticleInDomainTmp

@cuda.jit
def compactify_particles(pos, tile_index, cumulative_occupancy_flat, isParticleInDomain,
                         oldIndex):
    """
    """
    ip = cuda.grid(1)
    numParticles = pos.shape[0]
    # each thread takes care of a particle
    if (ip < numParticles):
        newPos = int(cumulative_occupancy_flat[ip])
        if (isParticleInDomain[ip] > 0):
            oldIndex[newPos - 1] = ip
            
@cuda.jit()
def apply_filter_optimized(oldIndex, pos, hsml, tile_index, 
                     start_index_for_tile, particles_per_tile, tile_widths,
                     variable, weights, offsets, npixs, center, widths, filter_lengths, 
                     smooth_var, filter_type):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    """
    # threadindex
    ip = cuda.grid(1)

    if (ip < oldIndex.shape[0]):
        oldIp = oldIndex[ip]
    
        # particle position
        xp = pos[oldIp, 0]
        yp = pos[oldIp, 1]
        zp = pos[oldIp, 2]

        # in theory we can have different filter lengths per particle
        # for the iterative scheme in Vazza this number is gradually increased
        # maybe this function needs to be reworked in that case...
        filter_length = filter_lengths[oldIp]

        ip_tile_x = tile_index[oldIp, 0]
        ip_tile_y = tile_index[oldIp, 1]
        ip_tile_z = tile_index[oldIp, 2]
    
        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0] 
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1] 
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2] 
    
        # tile_pos = tile_positions[tile_x, tile_y, tile_z]
        # tile_widths
        weight = 0.0
        weight_tmp = 0.0
        smoothVarRegister = 0.0
    
        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type == 1:
            filter_window = 4

        ip_tile_x_min = ip_tile_x - (- delta_x + \
                        filter_length + tile_widths[0] / 2) // tile_widths[0] 
        ip_tile_x_max = ip_tile_x + (delta_x +   \
                        filter_length + tile_widths[0] / 2) // tile_widths[0] 
    
        ip_tile_y_min = ip_tile_y - (- delta_y + \
                        filter_length + tile_widths[1] / 2) // tile_widths[1] 
        ip_tile_y_max = ip_tile_y + (delta_y +   \
                        filter_length + tile_widths[1] / 2) // tile_widths[1] 
    
        ip_tile_z_min = ip_tile_z - (- delta_z + \
                        filter_length + tile_widths[2] / 2) // tile_widths[2] 
        ip_tile_z_max = ip_tile_z + (delta_z +   \
                        filter_length + tile_widths[2] / 2) // tile_widths[2] 
    
        if filter_type == 0:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                          tile_x, tile_y, tile_z,
                                          delta_x, delta_y, delta_z,
                                          tile_widths, filter_length):

                            start_index = start_index_for_tile[tile_x,
                                                               tile_y, tile_z]
                            n_particles = particles_per_tile[tile_x,
                                                             tile_y, tile_z]
        
                            for ip_other in range(start_index, start_index + n_particles):
                                dist = distance(pos[oldIp], pos[ip_other])
                                if dist < filter_length:
                                    weight_tmp = 1.0 * weights[ip_other]
                                    weight += weight_tmp
                                    smoothVarRegister += variable[ip_other] * weight_tmp
    
        elif filter_type == 1:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                          tile_x, tile_y, tile_z,
                                          delta_x, delta_y, delta_z,
                                          tile_widths, filter_length):
                            start_index = start_index_for_tile[tile_x,
                                                               tile_y, tile_z]
                            n_particles = particles_per_tile[tile_x,
                                                             tile_y, tile_z]
        
                            for ip_other in range(start_index, start_index + n_particles):
                                dist = distance(pos[oldIp], pos[ip_other])
                                if dist < filter_length:
                                    weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                    weight += weight_tmp
                                    smoothVarRegister += variable[ip_other] * weight_tmp
        
        if weight > 0.:
            smooth_var[oldIp] = smoothVarRegister/weight

            

@cuda.jit()
def apply_filter(pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,
                 variable, weights, offsets, npixs, center, widths, filter_lengths, smooth_var, 
                 filter_type, hitsNeighbours, isParticleInDomain):
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

    xmin = center[0] - widths[0] / 2 - hsml[ip]
    xmax = center[0] + widths[0] / 2 + hsml[ip]

    ymin = center[1] - widths[1] / 2 - hsml[ip]
    ymax = center[1] + widths[1] / 2 + hsml[ip]

    zmin = center[2] - widths[2] / 2 - hsml[ip]
    zmax = center[2] + widths[2] / 2 + hsml[ip]

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

        isParticleInDomain[ip] += 1

        ip_tile_x = tile_index[ip, 0]
        ip_tile_y = tile_index[ip, 1]
        ip_tile_z = tile_index[ip, 2]

        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0] 
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1] 
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2] 

        # tile_pos = tile_positions[tile_x, tile_y, tile_z]
        # tile_widths
        weight = 0.0
        weight_tmp = 0.0

        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type == 1:
            filter_window = 4


        ip_tile_x_min = ip_tile_x - (- delta_x + \
                        filter_length + tile_widths[0] / 2) // tile_widths[0] 
        ip_tile_x_max = ip_tile_x + (delta_x +   \
                        filter_length + tile_widths[0] / 2) // tile_widths[0] 
    
        ip_tile_y_min = ip_tile_y - (- delta_y + \
                        filter_length + tile_widths[1] / 2) // tile_widths[1] 
        ip_tile_y_max = ip_tile_y + (delta_y +   \
                        filter_length + tile_widths[1] / 2) // tile_widths[1] 
    
        ip_tile_z_min = ip_tile_z - (- delta_z + \
                        filter_length + tile_widths[2] / 2) // tile_widths[2] 
        ip_tile_z_max = ip_tile_z + (delta_z +   \
                        filter_length + tile_widths[2] / 2) // tile_widths[2]

        if filter_type == 0:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                          tile_x, tile_y, tile_z,
                                          delta_x, delta_y, delta_z,
                                          tile_widths, filter_length):
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
                                    hitsNeighbours[ip] += 1

        elif filter_type == 1:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                          tile_x, tile_y, tile_z,
                                          delta_x, delta_y, delta_z,
                                          tile_widths, filter_length):
                            start_index = start_index_for_tile[tile_x,
                                                               tile_y, tile_z]
                            n_particles = particles_per_tile[tile_x,
                                                             tile_y, tile_z]
    
                            for ip_other in range(start_index, start_index + n_particles):
                                dist = distance(pos[ip], pos[ip_other])
                                if dist < filter_length:
                                    weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                    weight += weight_tmp
                                    smooth_var[ip] += variable[ip_other] * weight_tmp
                                    hitsNeighbours[ip] += 1
        if weight > 0.:
            smooth_var[ip] /= weight

@cuda.jit()
def apply_filter_spherical(pos, hsml, tile_index, start_index_for_tile,
                           particles_per_tile, spacings,
                           variable, weights, nSects, center, rMin, rMax, 
                           _rMin, filter_lengths, smooth_var, filter_type, 
                            hitsNeighbours, isParticleInDomain):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    rMin, rMax are the domain computational boundaries
    chosen by the user
    _rMin, _rMax are the lower and upper limits of 
    the radial grid (computed by SphericalTiling)
    _rMin < rMin (not exactly...)
    _rMax > rMax
    """
    # threadindex
    ip = cuda.grid(1)

    # particle position
    xp = pos[ip, 0]
    yp = pos[ip, 1]
    zp = pos[ip, 2]

    rad2 = (xp - center[0])**2 + \
           (yp - center[1])**2 + \
           (zp - center[2])**2

    rp = math.sqrt(rad2)
    phi = math.atan2(yp - center[1], xp - center[0]) % (2.0*math.pi)
    theta = math.acos( (zp - center[2]) / rp)
    #cylindrical radius
    cylRadius = rp * math.sin(theta)
    
    if (rMin > hsml[ip]):
        rad2Min = (rMin - hsml[ip])**2
    else:
        rad2Min = 0.0
    rad2Max = (rMax + hsml[ip])**2

    # Check if this cell/particle is inside domain
    inside_domain = False
    if (rad2 > rad2Min) and (rad2 < rad2Max):
        inside_domain = True

    # in theory we can have different filter lengths per particle
    # for the iterative scheme in Vazza this number is gradually increased
    # maybe this function needs to be reworked in that case...
    filter_length = filter_lengths[ip]

    radSpacing, phiSpacing, theSpacing = spacings
    nSectRad, nSectPhi, nSectThe = nSects
    nSectRad = int(nSectRad)
    nSectPhi = int(nSectPhi)
    nSectThe = int(nSectThe)
    
    if inside_domain:
        
        isParticleInDomain[ip] += 1

        ip_tile_rad = tile_index[ip, 0]
        ip_tile_phi = tile_index[ip, 1]
        ip_tile_the = tile_index[ip, 2]

        weight = 0.0
        weight_tmp = 0.0

        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type == 1:
            filter_window = 4

        # when the particle is far away from the z axis so that the filter
        # search radius does not overlap with it
        if (filter_length < cylRadius):
            delta_phi = math.asin((filter_length) / cylRadius)
            # tricky situations: 
            # 1: when phi - delta_phi < 0 or
            # 2: phi + delta_phi > 2 \pi
            # case 1: ip_tile_ip_min is negative (but with the 
            # appropriate value) and the range( , ) works as intended
            ip_tile_phi_min = int((phi - delta_phi) // phiSpacing)
            # case 2: we shift both phi_min and phi_max to the 
            # interval [-ip_tile_phi_min - numSectPhi (<0), 
            # ip_tile_phi_max - numSectPhi]
            # This leads back to case 1.
            ip_tile_phi_max = int((phi + delta_phi) // phiSpacing)
            if (phi + delta_phi > 2.0*math.pi):
                ip_tile_phi_min -= nSectPhi
                ip_tile_phi_max -= nSectPhi

            delta_theta = math.asin((filter_length) / rp)
            # when filter_length < cylRadius
            # theta +/- delta_theta is always well-behaved
            ip_tile_the_min = int((theta - delta_theta) // theSpacing)
            ip_tile_the_max = int((theta + delta_theta) // theSpacing)            
        # when the particle search radius overlaps with z axis
        # (cylRadius < filter_length)
        else:
            # need to search all the azimuthal sectors
            ip_tile_phi_min = 0
            ip_tile_phi_max = nSectPhi - 1
            # regarding latitudinal range, three cases:
            # case 1. particle+search radius is entirely in z>0 midplane
            # case 2. particle+search radius is entirely in z<0 midplane
            # case 3. particle+search radius overlaps with origin 
            # for case 3. the latitudinal tiles go from 0 to nSectThe -1
            ip_tile_the_min = 0
            ip_tile_the_max = nSectThe - 1
            if (( zp - center[2] ) > filter_length):
                # case 1. search latitudinal tiles from 0 to ip_tile_the_max
                delta_theta = math.asin((filter_length) / rp)
                ip_tile_the_max = int((theta + delta_theta) // theSpacing)
            elif (- (zp - center[2]) > filter_length):
                delta_theta = math.asin((filter_length) / rp)
                ip_tile_the_min = int((theta - delta_theta) // theSpacing)
                # case 2. search latitudinal tiles from ip_tile_the_min
                # to nSectThe - 1
            
        # the radial tile range selection actually is in 
        # common for both cases 
        # when filter_length < cylRadius
        # and when filter_length >= cylRadius
        # we have two cases: 
        # case 1. the ball overlaps with the origin
        # case 2. the ball does not overlap with the origin
        # both can be covered by the following
        delta_rad = filter_length
        ip_tile_rad_min = 0
        if (rp - delta_rad > _rMin):
            ip_tile_rad_min = int((math.log10(rp - delta_rad) - \
                               math.log10(_rMin) ) // radSpacing)
        ip_tile_rad_max = int((math.log10(rp + delta_rad) - \
                           math.log10(_rMin) ) // radSpacing)
            
        
        if filter_type == 0:
            for tile_rad in range(ip_tile_rad_min, ip_tile_rad_max + 1):
                for tile_phi in range(ip_tile_phi_min, ip_tile_phi_max + 1):
                    for tile_the in range(ip_tile_the_min, ip_tile_the_max + 1):

                        start_index = int(start_index_for_tile[tile_rad,
                                                           tile_phi, tile_the])
                        n_particles = int(particles_per_tile[tile_rad,
                                                           tile_phi, tile_the])

                        for ip_other in range(start_index, start_index + n_particles):
                            dist = distance(pos[ip], pos[ip_other])
                            if dist < filter_length:
                                weight_tmp = 1.0 * weights[ip_other]
                                weight += weight_tmp
                                smooth_var[ip] += variable[ip_other] * weight_tmp
                                hitsNeighbours[ip] += 1

        elif filter_type == 1:
            for tile_rad in range(ip_tile_rad_min, ip_tile_rad_max + 1):
                for tile_phi in range(ip_tile_phi_min, ip_tile_phi_max + 1):
                    for tile_the in range(ip_tile_the_min, ip_tile_the_max + 1):

                        start_index = int(start_index_for_tile[tile_rad,
                                                           tile_phi, tile_the])
                        n_particles = int(particles_per_tile[tile_rad,
                                                           tile_phi, tile_the])

                        for ip_other in range(start_index, start_index + n_particles):
                            dist = distance(pos[ip], pos[ip_other])
                            if dist < filter_length:
                                weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                weight += weight_tmp
                                smooth_var[ip] += variable[ip_other] * weight_tmp
                                hitsNeighbours[ip] += 1
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

    def __init__(self, snap, center, widths, orientation=None, search_radius=None,
                 npix=128, threadsperblock=256, tilingType='cartesian', numPhi=-1, 
                 numTheta=-1, rMin=-1.0, rMax=-1.0):
        """
        If spherical=True, npix is the number of intervals in the radial direction
        in the phi and theta direction we have npix, and npix/2 intervals
        by default
        """

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
        self.hsml = 2.0 * np.cbrt((self.snap["0_Volume"]) / (4.0 * np.pi / 3.0))

        if pa.settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        if search_radius is None:
            search_radius = 10.0 * self.hsml

        if hasattr(search_radius, 'unit'):
            self.search_radius = search_radius.copy
            assert search_radius.unit == code_length.unit, 'this restriction applies'
        elif pa.settings.use_units:
            self.search_radius = np.array(search_radius) * code_length
        else:
            self.search_radius = np.array(search_radius)

        # if max_search_radius is None:
        #     max_search_radius = 0.2 * np.max(self.widths)

        # if hasattr(max_search_radius, 'unit'):
        #     self.max_search_radius = max_search_radius.copy
        #     assert max_search_radius.unit == code_length.unit, 'this restriction applies'
        # elif pa.settings.use_units:
        #     self.max_search_radius = np.array(max_search_radius) * code_length
        # else:
        #     self.max_search_radius = np.array(max_search_radius)

        # tilingType = 'cartesian'
        # if spherical:
        #     tilingType = 'spherical'

        if (tilingType == 'cartesian'):
            self._do_region_selection()
        elif (tilingType == 'spherical'):
            self._do_region_selection_spherical()

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
                                        nRadial=128, nPhi=128, nTheta=64,
                                        threadsperblock=256)

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
        thickness = self.hsml 
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

    def _apply_filter_gpu(self, variable_str, weight, filter_type):
        pos = self.gpu_variables['pos']
        hsml = self.gpu_variables['hsml']
        tile_index = self.tile.tile_index
        start_index_for_tile = self.tile.start_index_for_tile
        particles_per_tile = self.tile.particles_per_tile
        variable = self.gpu_variables[variable_str]
        center = self.gpu_variables['center']
        offsets = self.tile.off_sets
        
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
        
        
        filter_lengths = self.gpu_variables['filter_lengths']
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        smooth_var = cp.zeros_like(variable)
        hitsNeighbours = cp.zeros(variable.shape,dtype="int")
        isParticleInDomain = cp.zeros(variable.shape,dtype="int")

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        if self.cartesian:
            apply_filter[self.blocks_1d, self.threadsperblock](pos, hsml, tile_index, start_index_for_tile,
                                                           particles_per_tile, tile_widths,
                                                           variable, weights, offsets, npixs, center, widths, 
                                                           filter_lengths, smooth_var, filter_type, hitsNeighbours,
                                                              isParticleInDomain)
        elif self.spherical:
            apply_filter_spherical[self.blocks_1d, self.threadsperblock](pos, hsml, tile_index, start_index_for_tile,
                                                           particles_per_tile, spacings,
                                                           variable, weights, nSects, center, rMin, rMax, _rMin,
                                                           filter_lengths, smooth_var, filter_type, hitsNeighbours,
                                                                             isParticleInDomain)
        self.hitsNeighbours = hitsNeighbours
        self.isParticleInDomainUnSorted = isParticleInDomain[self.tile.unsort_index]
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
                       shared_mem=False, Nmax=64, optimized=False):
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
            if (np.max(filter_length[self.indicesFirstPass]) > self.max_search_radius):
                raise RuntimeError('The chosen filter length is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self._send_variable_to_gpu(filter_length, gpu_key='filter_lengths')
        else:
            if (filter_length > self.max_search_radius):
                raise RuntimeError('The chosen filter length is larger than the \
                maximum search radius. This would cause searching for cells that \
                have not been moved to the GPU. To solve this decrease \
                the filter length or increase the search radius accordingly')
            self.gpu_variables['filter_lengths'] = cp.ones(self.Np) * filter_length

        # Do the filtering
        if not shared_mem:
            if not iterative:
                if optimized:
                    smooth_variable = self._apply_filter_gpu_optimized(variable_str, weight, filter_type)
                else: 
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


    def _apply_filter_gpu_optimized(self, variable_str, weight, filter_type):
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
        if filter_type == "mean":
            filter_type = 0
        elif filter_type == "gaussian":
            filter_type = 1
        

        if cp.max(filter_lengths) > self.extra_layer_thickness_value:
            err_msg = f"{cp.max(filter_lengths)} is larger than {self.extra_layer_thickness}"
            raise RuntimeError(err_msg)

        if weight is not None:
            weights = self.gpu_variables[weight]
        else:
            weights = cp.ones_like(variable)

        isParticleInDomain = cp.zeros(pos.shape[0])
        
        check_particle[self.blocks_1d, self.threadsperblock](pos, hsml, center, widths, isParticleInDomain)
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

        apply_filter_optimized[blocks_1d, self.threadsperblock](oldIndex, 
                                                          pos, hsml, tile_index, 
                                                          start_index_for_tile,
                                                           particles_per_tile, tile_widths,
                                                           variable, weights, offsets, npixs, center, widths, 
                                                          filter_lengths, smooth_var, filter_type)

        return cp.asnumpy(smooth_var[self.tile.unsort_index])

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        del self.gpu_variables
        # cp._default_memory_pool.free_all_blocks()
