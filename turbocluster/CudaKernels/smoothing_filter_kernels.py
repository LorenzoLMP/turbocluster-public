import cupy as cp
from numba import cuda
from .generic_kernels import *


@cuda.jit()
def apply_filter_optimized(oldIndex, pos, hsml, tile_index,
                           start_index_for_tile, particles_per_tile, tile_widths,
                           variable, weights, offsets, npixs, center, widths, filter_lengths,
                           smooth_var, filter_type, hitsNeighbours, isParticleInDomain,
                           iterativeFilter, hasConverged, numIterations, filter_lengths_out, multiplier,
                           max_filter_length):
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

        VarRegister = variable[oldIp]

        if (iterativeFilter == 0):  # no iterative
            filter_length = filter_lengths[oldIp]
        else:  # iterative
            # iterative scheme ~Vazza+2012
            # filter length is gradually increased
            # let's start from twice the radius
            filter_length = 2.0 * hsml[oldIp]
            # filter_length = 0.1 * filter_lengths[oldIp]
            # if (0.1 * filter_lengths[oldIp] < 1.0 * hsml[oldIp]):
            #     filter_length = 1.0 * hsml[oldIp]
            # let's start from 10% of the filter_length or 1 x hsml, whichever is largest

        # increase by one radius
        filterIncrease = hsml[oldIp]  # it is additive factor (?)
        # need to make sure that with max_filter_length it does not go
        # beyond region loaded on GPU

        hasIterationConverged = False

        turbFieldOld = 0.0
        turbFieldNew = 0.0
        toleranceParam = 0.05

        numInteractingPartOld = 0
        numInteractingPartNew = 0

        ip_tile_x = tile_index[oldIp, 0]
        ip_tile_y = tile_index[oldIp, 1]
        ip_tile_z = tile_index[oldIp, 2]

        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0]
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1]
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2]

        filter_window = 1
        # for gaussian filter we go up to 4*sigma
        # (sigma = filter_length) distance from the source particle
        if filter_type > 0:
            filter_window = multiplier

        numIter = 0
        while (filter_length <= max_filter_length and not hasIterationConverged):
            # the idea is to use this while loop both for iterative and non-iterative case
            # if non-iterative we exit after one go

            smoothVarRegister = 0.0
            normalization = 0.0
            weight_tmp = 0.0
            numInteractingPartNew = 0
            numSearchedPart = 0

            ####################################
            # code to check what are the tiles that can overlap
            # with the particle filter_length
            ####################################

            ip_tile_x_min = ip_tile_x - (- delta_x
                                         + filter_window * filter_length + tile_widths[0] / 2) // tile_widths[0]
            ip_tile_x_max = ip_tile_x + (delta_x
                                         + filter_window * filter_length + tile_widths[0] / 2) // tile_widths[0]

            ip_tile_y_min = ip_tile_y - (- delta_y
                                         + filter_window * filter_length + tile_widths[1] / 2) // tile_widths[1]
            ip_tile_y_max = ip_tile_y + (delta_y
                                         + filter_window * filter_length + tile_widths[1] / 2) // tile_widths[1]

            ip_tile_z_min = ip_tile_z - (- delta_z
                                         + filter_window * filter_length + tile_widths[2] / 2) // tile_widths[2]
            ip_tile_z_max = ip_tile_z + (delta_z
                                         + filter_window * filter_length + tile_widths[2] / 2) // tile_widths[2]

            ####################################
            # end of code to check overlap
            ####################################

            if filter_type == 0:
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):

                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    numSearchedPart += 1
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:
                                        # smoothing kernel
                                        weight_tmp = sphere_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        normalization += weight_tmp
                                        smoothVarRegister += variable[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            elif filter_type == 1:
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    numSearchedPart += 1
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:
                                        # smoothing kernel
                                        weight_tmp = gaussian_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        normalization += weight_tmp
                                        smoothVarRegister += variable[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            elif filter_type == 2:  # mexican hat filter
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    numSearchedPart += 1
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:
                                        # smoothing kernel
                                        weight_tmp = mexican_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        # for the mexican filter it is a bit different
                                        # the normalization uses the sphere kernel
                                        normalization += (weights[ip_other]
                                                          * (4. / 3.) * cp.pi * hsml[ip_other]**3
                                                          * sphere_kernel(dist, filter_window * filter_length))
                                        smoothVarRegister += variable[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            smoothVarRegister /= normalization

            ################################
            # part to check convergence of iterative filter
            ################################

            if (iterativeFilter == 0):  # no iterative
                hasIterationConverged = True
            else:  # iterative
                turbFieldNew = VarRegister - smoothVarRegister
                increase = math.fabs(turbFieldNew - turbFieldOld)
                # we declare the iteration converged when all the following are satisfied:
                # 1. the relative increase is less than the tolerance
                # 2. we have done at least one iteration (so we can compare old and new)
                # 3. the number of particles at the new iteration has increased
                # (if the number did not increase this would naturally cause 1. to be true)
                # 4. the number of particles it interacts with is > 8 (2 per dim, arbitrary)

                if (numIter > 1) and (increase <= toleranceParam * (0.0 + math.fabs(turbFieldOld))):
                    if (numInteractingPartNew > numInteractingPartOld):
                        if (numInteractingPartNew > 8):
                            hasIterationConverged = True
                            hasConverged[oldIp] = 1
                else:  # not converged
                    filter_length += filterIncrease
                    turbFieldOld = turbFieldNew
                    numIter += 1
                    numInteractingPartOld = numInteractingPartNew

            ################################
            # end of check convergence of iterative filter
            ################################

        smooth_var[oldIp] = smoothVarRegister
        hitsNeighbours[oldIp] = numInteractingPartNew
        # if not iterative we are going to be recycling the
        # hasConverged array to store how many particles it has searched
        if (iterativeFilter == 0):  # not iterative
            hasConverged[oldIp] = numSearchedPart
        if (iterativeFilter == 1):  # iterative
            filter_lengths_out[oldIp] = filter_length
            numIterations[oldIp] = numIter
            if not hasIterationConverged:
                # this is to adjust for the last iteration
                filter_lengths_out[oldIp] -= filterIncrease
                numIterations[oldIp] -= 1


@cuda.jit()
def apply_filter_optimized_vector(oldIndex, pos, hsml, tile_index,
                                  start_index_for_tile, particles_per_tile, tile_widths,
                                  variable_x, variable_y, variable_z, weights, offsets, npixs, center, widths, filter_lengths,
                                  smooth_var_x, smooth_var_y, smooth_var_z, filter_type, hitsNeighbours, isParticleInDomain,
                                  iterativeFilter, hasConverged, numIterations, filter_lengths_out, multiplier,
                                  max_filter_length):
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

        VarRegister_x = variable_x[oldIp]
        VarRegister_y = variable_y[oldIp]
        VarRegister_z = variable_z[oldIp]

        if (iterativeFilter == 0):  # no iterative
            filter_length = filter_lengths[oldIp]
        else:  # iterative
            # iterative scheme ~Vazza+2012
            # filter length is gradually increased
            # let's start from twice the radius
            filter_length = 2.0 * hsml[oldIp]
            # filter_length = 0.1 * filter_lengths[oldIp]
            # if (0.1 * filter_lengths[oldIp] < 1.0 * hsml[oldIp]):
            #     filter_length = 1.0 * hsml[oldIp]
            # let's start from 10% of the filter_length or 1 x hsml, whichever is largest

        # increase by one radius
        filterIncrease = hsml[oldIp]  # it is additive factor (?)
        # max_filter_length = 10.0*filter_lengths[oldIp]
        # need to make sure that with max_filter_length it does not go
        # beyond region loaded on GPU

        hasIterationConverged = False

        turbFieldOld_x, turbFieldOld_y, turbFieldOld_z = 0.0, 0.0, 0.0
        turbFieldNew_x, turbFieldNew_y, turbFieldNew_z = 0.0, 0.0, 0.0
        toleranceParam = 0.05

        numInteractingPartOld = 0
        numInteractingPartNew = 0

        ip_tile_x = tile_index[oldIp, 0]
        ip_tile_y = tile_index[oldIp, 1]
        ip_tile_z = tile_index[oldIp, 2]

        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0]
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1]
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2]

        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type > 0:
            filter_window = multiplier

        numIter = 0
        while (filter_length <= max_filter_length and not hasIterationConverged):
            # the idea is to use this while loop both for iterative and non-iterative case
            # if non-iterative we exit after one go

            smoothVarRegister_x, smoothVarRegister_y, smoothVarRegister_z = 0.0, 0.0, 0.0
            normalization = 0.0
            weight_tmp = 0.0
            numInteractingPartNew = 0

            ####################################
            # code to check what are the tiles that can overlap
            # with the particle filter_length
            ####################################

            ip_tile_x_min = ip_tile_x - (- delta_x
                                         + filter_window * filter_length + tile_widths[0] / 2) // tile_widths[0]
            ip_tile_x_max = ip_tile_x + (delta_x
                                         + filter_window * filter_length + tile_widths[0] / 2) // tile_widths[0]

            ip_tile_y_min = ip_tile_y - (- delta_y
                                         + filter_window * filter_length + tile_widths[1] / 2) // tile_widths[1]
            ip_tile_y_max = ip_tile_y + (delta_y
                                         + filter_window * filter_length + tile_widths[1] / 2) // tile_widths[1]

            ip_tile_z_min = ip_tile_z - (- delta_z
                                         + filter_window * filter_length + tile_widths[2] / 2) // tile_widths[2]
            ip_tile_z_max = ip_tile_z + (delta_z
                                         + filter_window * filter_length + tile_widths[2] / 2) // tile_widths[2]

            ####################################
            # end of code to check overlap
            ####################################

            if filter_type == 0:
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):

                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:

                                        # smoothing kernel
                                        weight_tmp = sphere_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        normalization += weight_tmp

                                        smoothVarRegister_x += variable_x[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_y += variable_y[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_z += variable_z[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            elif filter_type == 1:
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:
                                        # smoothing kernel
                                        weight_tmp = gaussian_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        normalization += weight_tmp

                                        smoothVarRegister_x += variable_x[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_y += variable_y[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_z += variable_z[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            elif filter_type == 2:  # mexican-hat filter
                for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                    for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                        for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                            if check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                                              tile_x, tile_y, tile_z,
                                              delta_x, delta_y, delta_z,
                                              tile_widths, filter_window * filter_length):
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]

                                for ip_other in range(start_index, start_index + n_particles):
                                    dist = distance(
                                        (xp, yp, zp), pos[ip_other])
                                    if dist < filter_window * filter_length:
                                        # smoothing kernel
                                        weight_tmp = mexican_kernel(
                                            dist, filter_length)
                                        # optional weight (e.g. mass, density, ...)
                                        weight_tmp *= weights[ip_other]
                                        # volume of voronoi cell
                                        weight_tmp *= (4. / 3.) * \
                                            cp.pi * hsml[ip_other]**3
                                        # normalization of the integral
                                        # for the mexican filter it is a bit different
                                        # the normalization uses the sphere kernel
                                        normalization += (weights[ip_other]
                                                          * (4. / 3.) * cp.pi * hsml[ip_other]**3
                                                          * sphere_kernel(dist, filter_window * filter_length))

                                        smoothVarRegister_x += variable_x[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_y += variable_y[ip_other] * \
                                            weight_tmp
                                        smoothVarRegister_z += variable_z[ip_other] * \
                                            weight_tmp
                                        numInteractingPartNew += 1

            smoothVarRegister_x /= normalization
            smoothVarRegister_y /= normalization
            smoothVarRegister_z /= normalization

            ################################
            # part to check convergence of iterative filter
            ################################

            if (iterativeFilter == 0):  # no iterative
                hasIterationConverged = True
            else:  # iterative
                turbFieldNew_x = VarRegister_x - smoothVarRegister_x
                turbFieldNew_y = VarRegister_y - smoothVarRegister_y
                turbFieldNew_z = VarRegister_z - smoothVarRegister_z

                # here we can decide on what to base the convergence
                # criterion: e.g., the norm of the fluctuation vector, or the square, etc...
                # this is the L2 norm convergence
                norm_Old = math.sqrt(turbFieldOld_x**2
                                     + turbFieldOld_y**2 + turbFieldOld_z**2)
                # relIncrease =  math.sqrt((turbFieldNew_x - turbFieldOld_x)**2 + (turbFieldNew_y - turbFieldOld_y)**2 + \
                # (turbFieldNew_z - turbFieldOld_z)**2 ) / norm_Old
                increase = math.sqrt((turbFieldNew_x - turbFieldOld_x)**2
                                     + (turbFieldNew_y - turbFieldOld_y)**2
                                     + (turbFieldNew_z - turbFieldOld_z)**2)

                # we declare the iteration converged when all the following are satisfied:
                # 1. the relative increase is less than the tolerance
                # 2. we have done at least one iteration (so we can compare old and new)
                # 3. the number of particles at the new iteration has increased
                # (if the number did not increase this would naturally cause 1. to be true)
                # 4. the number of particles it interacts with is > 8 (2 per dim, arbitrary)

                if (numIter > 1) and (increase <= toleranceParam * (0.0 + math.fabs(norm_Old))):
                    if (numInteractingPartNew > numInteractingPartOld):
                        if (numInteractingPartNew > 8):
                            hasIterationConverged = True
                            hasConverged[oldIp] = 1
                # if (relIncrease <= toleranceParam and numIter > 0 and numInteractingPartNew > numInteractingPartOld):
                #     hasIterationConverged = True
                #     hasConverged[oldIp] = 1
                else:  # not converged
                    filter_length += filterIncrease
                    turbFieldOld_x = turbFieldNew_x
                    turbFieldOld_y = turbFieldNew_y
                    turbFieldOld_z = turbFieldNew_z
                    numIter += 1
                    numInteractingPartOld = numInteractingPartNew

            ################################
            # end of check convergence of iterative filter
            ################################

        smooth_var_x[oldIp] = smoothVarRegister_x
        smooth_var_y[oldIp] = smoothVarRegister_y
        smooth_var_z[oldIp] = smoothVarRegister_z

        hitsNeighbours[oldIp] = numInteractingPartNew
        if (iterativeFilter == 1):  # iterative
            filter_lengths_out[oldIp] = filter_length
            numIterations[oldIp] = numIter
            if not hasIterationConverged:
                # this is to adjust for the last iteration
                filter_lengths_out[oldIp] -= filterIncrease
                numIterations[oldIp] -= 1


@cuda.jit()
def compute_max_hsml(hsml, compactGrid, maxHsmlPerBlock):
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
            hsmlMaxTmp = hsml[particleIp] if (
                hsml[particleIp] > hsmlMaxTmp) else hsmlMaxTmp

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
