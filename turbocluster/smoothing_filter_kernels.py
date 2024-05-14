import cupy as cp
from numba import cuda
from .generic_kernels import *

@cuda.jit(lineinfo=True)
def apply_filter_optimized(oldIndex, pos, hsml, tile_index, 
                     start_index_for_tile, particles_per_tile, tile_widths,
                     variable, weights, offsets, npixs, center, widths, filter_lengths, 
                     smooth_var, filter_type, hitsNeighbours, isParticleInDomain, 
                     iterativeFilter, hasConverged, numIterations, filter_lengths_out):
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

        oldVarRegister = variable[oldIp]

        if (iterativeFilter == 0): # no iterative
            filter_length = filter_lengths[oldIp]
        else: #iterative
            # iterative scheme ~Vazza+2012 
            # filter length is gradually increased
            filter_length = 0.1 * filter_lengths[oldIp] 
            if (0.1 * filter_lengths[oldIp] < 1.0 * hsml[oldIp]):  
                filter_length = 1.0 * hsml[oldIp] 
            # let's start from 10% of the filter_length or 1 x hsml, whichever is largest
            
        filterIncrease = 0.5*hsml[oldIp] # it is additive factor (?)
        max_filter_length = 5.0*filter_lengths[oldIp] 
        # need to make sure that with max_filter_length it does not go
        # beyond region loaded on GPU
        
        hasIterationConverged = False
    
        turbFieldOld = 0.0
        turbFieldNew = 0.0
        toleranceParam = 0.1
    
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
        if filter_type == 1:
            filter_window = 4

        numIter = 0
        while (filter_length <= max_filter_length and not hasIterationConverged):
            # the idea is to use this while loop both for iterative and non-iterative case
            # if non-iterative we exit after one go
            
            smoothVarRegister = 0.0
            weight = 0.0
            weight_tmp = 0.0
            numInteractingPartNew = 0

            ####################################
            # code to check what are the tiles that can overlap
            # with the particle filter_length
            ####################################

            
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
                                              tile_widths, filter_length):
    
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]
            
                                for ip_other in range(start_index, start_index + n_particles):
                                    dist = distance((xp, yp, zp), pos[ip_other])
                                    if dist < filter_length:
                                        weight_tmp = 1.0 * weights[ip_other]
                                        weight += weight_tmp
                                        smoothVarRegister += variable[ip_other] * weight_tmp
                                        numInteractingPartNew += 1
        
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
                                    dist = distance((xp, yp, zp), pos[ip_other])
                                    if dist < filter_length:
                                        weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                        weight += weight_tmp
                                        smoothVarRegister += variable[ip_other] * weight_tmp
                                        numInteractingPartNew += 1
            
            if weight > 0.:
                smoothVarRegister /= weight

            ################################
            # part to check convergence of iterative filter
            ################################
            
            if (iterativeFilter == 0): # no iterative
                hasIterationConverged = True
            else: # iterative
                turbFieldNew = oldVarRegister - smoothVarRegister
                relIncrease =  abs(turbFieldNew - turbFieldOld) / abs(turbFieldOld)
                # we declare the iteration converged when all the following are satisfied:
                # 1. the relative increase is less than the tolerance
                # 2. we have done at least one iteration (so we can compare old and new)
                # 3. the number of particles at the new iteration has increased
                # (if the number did not increase this would naturally cause 1. to be true)
                if (relIncrease <= toleranceParam and numIter > 0 and numInteractingPartNew > numInteractingPartOld):
                    hasIterationConverged = True
                    hasConverged[oldIp] = 1
                else: # not converged
                    filter_length += filterIncrease
                    turbFieldOld = turbFieldNew
                    numIter += 1
                    numInteractingPartOld = numInteractingPartNew

            ################################
            # end of check convergence of iterative filter
            ################################

        smooth_var[oldIp] = smoothVarRegister
        hitsNeighbours[oldIp] = numInteractingPartNew
        if (iterativeFilter == 1): # iterative
            filter_lengths_out[oldIp] = filter_length
            numIterations[oldIp] = numIter
            if not hasIterationConverged:
                # this is to adjust for the last iteration
                filter_lengths_out[oldIp] -= filterIncrease
                numIterations[oldIp] -= 1
                

            

@cuda.jit(lineinfo=True)
def apply_filter(pos, hsml, tile_index, start_index_for_tile, particles_per_tile, tile_widths,
                 variable, weights, offsets, npixs, center, widths, filter_lengths, smooth_var, 
                 filter_type, hitsNeighbours, isParticleInDomain, iterativeFilter, hasConverged, 
                 numIterations, filter_lengths_out):
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

    varRegister = variable[ip]

    xmin = center[0] - widths[0] / 2 - hsml[ip]
    xmax = center[0] + widths[0] / 2 + hsml[ip]

    ymin = center[1] - widths[1] / 2 - hsml[ip]
    ymax = center[1] + widths[1] / 2 + hsml[ip]

    zmin = center[2] - widths[2] / 2 - hsml[ip]
    zmax = center[2] + widths[2] / 2 + hsml[ip]

    
    if (iterativeFilter == 0): # no iterative
        filter_length = filter_lengths[ip]
    else: #iterative
        # iterative scheme ~Vazza+2012 
        # filter length is gradually increased
        filter_length = 0.1 * filter_lengths[ip] 
        if (0.1 * filter_lengths[ip] < 1.0 * hsml[ip]):  
            filter_length = 1.0 * hsml[ip] 
        # let's start from 10% of the filter_length or 1 x hsml, whichever is largest
        
    filterIncrease = 0.5*hsml[ip] # it is additive factor (?)
    max_filter_length = 5.0*filter_lengths[ip] 
    # need to make sure that with max_filter_length it does not go
    # beyond region loaded on GPU
    
    hasIterationConverged = False

    turbFieldOld = 0.0
    turbFieldNew = 0.0
    toleranceParam = 0.1

    numInteractingPartOld = 0
    numInteractingPartNew = 0
    

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

        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type == 1:
            filter_window = 4

        numIter = 0
        while (filter_length <= max_filter_length and not hasIterationConverged):
            # the idea is to use this while loop both for iterative and non-iterative case
            # if non-iterative we exit after one go
            scratchSmooth = 0.0
            weight = 0.0
            weight_tmp = 0.0
            numInteractingPartNew = 0

            ####################################
            # code to check what are the tiles that can overlap
            # with the particle filter_length
            ####################################
            
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
                                              tile_widths, filter_length):
                                start_index = start_index_for_tile[tile_x,
                                                                   tile_y, tile_z]
                                n_particles = particles_per_tile[tile_x,
                                                                 tile_y, tile_z]
                                
        
                                for ip_other in range(start_index, start_index + n_particles):
                                    dist = distance((xp, yp, zp), pos[ip_other])
                                    if dist < filter_length:
                                        weight_tmp = 1.0 * weights[ip_other]
                                        weight += weight_tmp
                                        scratchSmooth += variable[ip_other] * weight_tmp
                                        numInteractingPartNew += 1
    
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
                                    dist = distance((xp, yp, zp), pos[ip_other])
                                    if dist < filter_length:
                                        weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                        weight += weight_tmp
                                        scratchSmooth += variable[ip_other] * weight_tmp
                                        numInteractingPartNew += 1
                                        
            if weight > 0.:
                scratchSmooth /= weight

            ################################
            # part to check convergence of iterative filter
            ################################
            
            if (iterativeFilter == 0): # no iterative
                hasIterationConverged = True
            else: # iterative
                turbFieldNew = varRegister - scratchSmooth
                relIncrease =  abs(turbFieldNew - turbFieldOld) / abs(turbFieldOld)
                # we declare the iteration converged when all the following are satisfied:
                # 1. the relative increase is less than the tolerance
                # 2. we have done at least one iteration (so we can compare old and new)
                # 3. the number of particles at the new iteration has increased
                # (if the number did not increase this would naturally cause 1. to be true)
                if (relIncrease <= toleranceParam and numIter > 0 and numInteractingPartNew > numInteractingPartOld):
                    hasIterationConverged = True
                    hasConverged[ip] = 1
                else: # not converged
                    filter_length += filterIncrease
                    turbFieldOld = turbFieldNew
                    numIter += 1
                    numInteractingPartOld = numInteractingPartNew

            ################################
            # end of check convergence of iterative filter
            ################################

        smooth_var[ip] = scratchSmooth
        hitsNeighbours[ip] = numInteractingPartNew
        if (iterativeFilter == 1): # iterative
            filter_lengths_out[ip] = filter_length
            numIterations[ip] = numIter
            if not hasIterationConverged:
                # this is to adjust for the last iteration
                filter_lengths_out[ip] -= filterIncrease
                numIterations[ip] -= 1

@cuda.jit(lineinfo=True)
def apply_filter_spherical(pos, hsml, tile_index, start_index_for_tile,
                           particles_per_tile, spacings,
                           variable, weights, nSects, center, rMin, rMax, 
                           _rMin, _rMax, filter_lengths, smooth_var, filter_type, 
                            hitsNeighbours, isParticleInDomain, typeGrid, power, 
                           iterativeFilter, hasConverged, numIterations, filter_lengths_out):
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

    varRegister = variable[ip]

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

    if (iterativeFilter == 0): # no iterative
        filter_length = filter_lengths[ip]
    else: #iterative
        # iterative scheme ~Vazza+2012
        # filter length is gradually increased
        filter_length = 0.1 * filter_lengths[ip]
        if (0.1 * filter_lengths[ip] < 1.0 * hsml[ip]):
            filter_length = 1.0 * hsml[ip]
        # let's start from 10% of the filter_length or 1 x hsml, whichever is largest

    filterIncrease = 0.5*hsml[ip] # it is additive factor (?)
    max_filter_length = 5.0*filter_lengths[ip]
    # need to make sure that with max_filter_length it does not go
    # beyond region loaded on GPU

    hasIterationConverged = False

    turbFieldOld = 0.0
    turbFieldNew = 0.0
    toleranceParam = 0.1

    numInteractingPartOld = 0
    numInteractingPartNew = 0

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

        filter_window = 1
        # for gaussian filter the sigma is  1/4 of the
        # filter_length of the source particle
        if filter_type == 1:
            filter_window = 4

        numIter = 0
        while (filter_length <= max_filter_length and not hasIterationConverged):
            # the idea is to use this while loop both for iterative and non-iterative case
            # if non-iterative we exit after one go
            scratchSmooth = 0.0
            weight = 0.0
            weight_tmp = 0.0
            numInteractingPartNew = 0

            ####################################
            # code to check what are the tiles that can overlap
            # with the particle filter_length
            ####################################
            
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
                # ip_tile_rad_min = int((math.log10(rp - delta_rad) - \
                #                    math.log10(_rMin) ) // radSpacing)
                ip_tile_rad_min = find_tile_radial(rp - delta_rad, radSpacing, _rMin, _rMax, typeGrid, power)
            # ip_tile_rad_max = int((math.log10(rp + delta_rad) - \
            #                    math.log10(_rMin) ) // radSpacing)
            ip_tile_rad_max = find_tile_radial(rp + delta_rad, radSpacing, _rMin, _rMax, typeGrid, power)

            ####################################
            # end of code to check overlap
            ####################################
            
            if filter_type == 0:
                for tile_rad in range(ip_tile_rad_min, ip_tile_rad_max + 1):
                    for tile_phi in range(ip_tile_phi_min, ip_tile_phi_max + 1):
                        for tile_the in range(ip_tile_the_min, ip_tile_the_max + 1):
    
                            start_index = int(start_index_for_tile[tile_rad,
                                                               tile_phi, tile_the])
                            n_particles = int(particles_per_tile[tile_rad,
                                                               tile_phi, tile_the])
    
                            for ip_other in range(start_index, start_index + n_particles):
                                dist = distance((xp, yp, zp), pos[ip_other])
                                if dist < filter_length:
                                    weight_tmp = 1.0 * weights[ip_other]
                                    weight += weight_tmp
                                    scratchSmooth += variable[ip_other] * weight_tmp
                                    numInteractingPartNew += 1
    
            elif filter_type == 1:
                for tile_rad in range(ip_tile_rad_min, ip_tile_rad_max + 1):
                    for tile_phi in range(ip_tile_phi_min, ip_tile_phi_max + 1):
                        for tile_the in range(ip_tile_the_min, ip_tile_the_max + 1):
    
                            start_index = int(start_index_for_tile[tile_rad,
                                                               tile_phi, tile_the])
                            n_particles = int(particles_per_tile[tile_rad,
                                                               tile_phi, tile_the])
    
                            for ip_other in range(start_index, start_index + n_particles):
                                dist = distance((xp, yp, zp), pos[ip_other])
                                if dist < filter_length:
                                    weight_tmp = gaussian_kernel(dist, filter_length / filter_window) * weights[ip_other]
                                    weight += weight_tmp
                                    scratchSmooth += variable[ip_other] * weight_tmp
                                    numInteractingPartNew += 1
            if weight > 0.:
                scratchSmooth /= weight

            ################################
            # part to check convergence of iterative filter
            ################################
            
            if (iterativeFilter == 0): # no iterative
                hasIterationConverged = True
            else: # iterative
                turbFieldNew = varRegister - scratchSmooth
                relIncrease =  abs(turbFieldNew - turbFieldOld) / abs(turbFieldOld)
                # we declare the iteration converged when all the following are satisfied:
                # 1. the relative increase is less than the tolerance
                # 2. we have done at least one iteration (so we can compare old and new)
                # 3. the number of particles at the new iteration has increased
                # (if the number did not increase this would naturally cause 1. to be true)
                if (relIncrease <= toleranceParam and numIter > 0 and numInteractingPartNew > numInteractingPartOld):
                    hasIterationConverged = True
                    hasConverged[ip] = 1
                else: # not converged
                    filter_length += filterIncrease
                    turbFieldOld = turbFieldNew
                    numIter += 1
                    numInteractingPartOld = numInteractingPartNew

            ################################
            # end of check convergence of iterative filter
            ################################

        smooth_var[ip] = scratchSmooth
        hitsNeighbours[ip] = numInteractingPartNew
        if (iterativeFilter == 1): # iterative
            filter_lengths_out[ip] = filter_length
            numIterations[ip] = numIter
            if not hasIterationConverged:
                # this is to adjust for the last iteration
                filter_lengths_out[ip] -= filterIncrease
                numIterations[ip] -= 1


@cuda.jit(device=True, inline=True)
def find_tile_radial(particle_R, radSpacing, rMin, rMax, type, power):
    """
    rMin include the extra thickness (_rMin)
    rMax include the extra thickness (_rMax)
    type = 0 is log
    type = 1 is power law with "power" exponent
    """
    radTileIndex = 0
    if (type == 0):
        if (particle_R > rMin):
            radTileIndex = (math.log10(particle_R) - math.log10(rMin) ) // radSpacing
    elif (type == 1):
        radTileIndex = ( (particle_R - rMin)/(rMax - rMin) )**power // radSpacing
    return int(radTileIndex)

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
        


@cuda.jit(lineinfo=True)
def apply_filter_shared(compactGrid, pos, hsml, tile_index, start_index_for_tile, 
                        particles_per_tile, tile_widths, variable, weights, center, 
                        widths, npixs, filter_lengths, smooth_var, filter_type, 
                        hitsNeighbours, isParticleInDomain):
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

    if filter_type == 0:
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
                                    hitsNeighbours[idOwnParticle] += 1
                
                        cuda.syncthreads()
    
    elif filter_type == 1:
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
                                    weight_tmp = gaussian_kernel(dist, ownFilterLength / filter_window) \
                                                * neighParticleBuf[ip*5 + 3]
                                    # weight_tmp = gaussian_kernel(dist, ownFilterLength / filter_window) \
                                    #             * neighParticleBuf[ipShifted*5 + 3]
                                    weight += weight_tmp
                                    smoothVarRegister += neighParticleBuf[ip*5 + 4] * weight_tmp
                                    # smoothVarRegister += neighParticleBuf[ipShifted*5 + 4] * weight_tmp
                                    hitsNeighbours[idOwnParticle] += 1
                
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
                    isParticleInDomain[idOwnParticle] += 1
