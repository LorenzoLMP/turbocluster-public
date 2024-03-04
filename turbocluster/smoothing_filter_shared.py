import cupy as cp
from numba import cuda



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
