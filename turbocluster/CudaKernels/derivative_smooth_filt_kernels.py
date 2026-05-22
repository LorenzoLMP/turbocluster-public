import cupy as cp
from numba import cuda
from .generic_kernels import distance, check_distance, gradient_gaussian


@cuda.jit()
def apply_derivative(
    oldIndex,
    pos,
    hsml,
    tile_index,
    start_index_for_tile,
    particles_per_tile,
    tile_widths,
    variable,
    offsets,
    npixs,
    center,
    widths,
    filter_lengths,
    grad_x_var,
    grad_y_var,
    grad_z_var,
    filter_type,
    hitsNeighbours,
    isParticleInDomain,
    hasConverged,
    multiplier,
):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    """
    # threadindex
    ip = cuda.grid(1)

    if ip < oldIndex.shape[0]:
        oldIp = oldIndex[ip]

        # particle position
        xp = pos[oldIp, 0]
        yp = pos[oldIp, 1]
        zp = pos[oldIp, 2]

        filter_length = filter_lengths[oldIp]

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

        grad_x_VarRegister = 0.0
        grad_y_VarRegister = 0.0
        grad_z_VarRegister = 0.0

        numInteractingPartNew = 0
        numSearchedPart = 0

        ####################################
        # code to check what are the tiles that can overlap
        # with the particle filter_length
        ####################################

        ip_tile_x_min = (
            ip_tile_x
            - (-delta_x + filter_window * filter_length + tile_widths[0] / 2)
            // tile_widths[0]
        )
        ip_tile_x_max = (
            ip_tile_x
            + (delta_x + filter_window * filter_length + tile_widths[0] / 2)
            // tile_widths[0]
        )

        ip_tile_y_min = (
            ip_tile_y
            - (-delta_y + filter_window * filter_length + tile_widths[1] / 2)
            // tile_widths[1]
        )
        ip_tile_y_max = (
            ip_tile_y
            + (delta_y + filter_window * filter_length + tile_widths[1] / 2)
            // tile_widths[1]
        )

        ip_tile_z_min = (
            ip_tile_z
            - (-delta_z + filter_window * filter_length + tile_widths[2] / 2)
            // tile_widths[2]
        )
        ip_tile_z_max = (
            ip_tile_z
            + (delta_z + filter_window * filter_length + tile_widths[2] / 2)
            // tile_widths[2]
        )

        ####################################
        # end of code to check overlap
        ####################################

        # only derivative of Gaussian implemented for now
        if filter_type == 1:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(
                            ip_tile_x,
                            ip_tile_y,
                            ip_tile_z,
                            tile_x,
                            tile_y,
                            tile_z,
                            delta_x,
                            delta_y,
                            delta_z,
                            tile_widths,
                            filter_window * filter_length,
                        ):
                            start_index = start_index_for_tile[tile_x, tile_y, tile_z]
                            n_particles = particles_per_tile[tile_x, tile_y, tile_z]

                            for ip_other in range(
                                start_index, start_index + n_particles
                            ):
                                numSearchedPart += 1
                                dist = distance((xp, yp, zp), pos[ip_other])
                                if dist < filter_window * filter_length:
                                    # gradient of kernel
                                    grad_x, grad_y, grad_z = gradient_gaussian(
                                        (xp, yp, zp), pos[ip_other], dist, filter_length
                                    )
                                    # volume of voronoi cell
                                    volume = (4.0 / 3.0) * cp.pi * hsml[ip_other] ** 3
                                    grad_x_VarRegister += (
                                        variable[ip_other] * grad_x * volume
                                    )
                                    grad_y_VarRegister += (
                                        variable[ip_other] * grad_y * volume
                                    )
                                    grad_z_VarRegister += (
                                        variable[ip_other] * grad_z * volume
                                    )

                                    numInteractingPartNew += 1

        # smoothVarRegister /= normalization

        grad_x_var[oldIp] = grad_x_VarRegister
        grad_y_var[oldIp] = grad_y_VarRegister
        grad_z_var[oldIp] = grad_z_VarRegister

        hitsNeighbours[oldIp] = numInteractingPartNew
        # if not iterative we are going to be recycling the
        # hasConverged array to store how many particles it has searched
        hasConverged[oldIp] = numSearchedPart


@cuda.jit()
def apply_derivative_vector(
    oldIndex,
    pos,
    hsml,
    tile_index,
    start_index_for_tile,
    particles_per_tile,
    tile_widths,
    variable_x,
    variable_y,
    variable_z,
    offsets,
    npixs,
    center,
    widths,
    filter_lengths,
    grad_x_vec_x,
    grad_x_vec_y,
    grad_x_vec_z,
    grad_y_vec_x,
    grad_y_vec_y,
    grad_y_vec_z,
    grad_z_vec_x,
    grad_z_vec_y,
    grad_z_vec_z,
    filter_type,
    hitsNeighbours,
    isParticleInDomain,
    hasConverged,
    multiplier,
):
    """
    filter_lengths is an array of size pos.shape([0])
    type can be "mean" or "gaussian"
    """
    # threadindex
    ip = cuda.grid(1)

    if ip < oldIndex.shape[0]:
        oldIp = oldIndex[ip]

        # particle position
        xp = pos[oldIp, 0]
        yp = pos[oldIp, 1]
        zp = pos[oldIp, 2]

        filter_length = filter_lengths[oldIp]

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

        grad_x_Vec_x_Reg, grad_x_Vec_y_Reg, grad_x_Vec_z_Reg = 0.0, 0.0, 0.0
        grad_y_Vec_x_Reg, grad_y_Vec_y_Reg, grad_y_Vec_z_Reg = 0.0, 0.0, 0.0
        grad_z_Vec_x_Reg, grad_z_Vec_y_Reg, grad_z_Vec_z_Reg = 0.0, 0.0, 0.0

        numInteractingPartNew = 0
        numSearchedPart = 0

        ####################################
        # code to check what are the tiles that can overlap
        # with the particle filter_length
        ####################################

        ip_tile_x_min = (
            ip_tile_x
            - (-delta_x + filter_window * filter_length + tile_widths[0] / 2)
            // tile_widths[0]
        )
        ip_tile_x_max = (
            ip_tile_x
            + (delta_x + filter_window * filter_length + tile_widths[0] / 2)
            // tile_widths[0]
        )

        ip_tile_y_min = (
            ip_tile_y
            - (-delta_y + filter_window * filter_length + tile_widths[1] / 2)
            // tile_widths[1]
        )
        ip_tile_y_max = (
            ip_tile_y
            + (delta_y + filter_window * filter_length + tile_widths[1] / 2)
            // tile_widths[1]
        )

        ip_tile_z_min = (
            ip_tile_z
            - (-delta_z + filter_window * filter_length + tile_widths[2] / 2)
            // tile_widths[2]
        )
        ip_tile_z_max = (
            ip_tile_z
            + (delta_z + filter_window * filter_length + tile_widths[2] / 2)
            // tile_widths[2]
        )

        ####################################
        # end of code to check overlap
        ####################################

        # only derivative of Gaussian implemented for now
        if filter_type == 1:
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                        if check_distance(
                            ip_tile_x,
                            ip_tile_y,
                            ip_tile_z,
                            tile_x,
                            tile_y,
                            tile_z,
                            delta_x,
                            delta_y,
                            delta_z,
                            tile_widths,
                            filter_window * filter_length,
                        ):
                            start_index = start_index_for_tile[tile_x, tile_y, tile_z]
                            n_particles = particles_per_tile[tile_x, tile_y, tile_z]

                            for ip_other in range(
                                start_index, start_index + n_particles
                            ):
                                numSearchedPart += 1
                                dist = distance((xp, yp, zp), pos[ip_other])
                                if dist < filter_window * filter_length:
                                    # gradient of kernel
                                    grad_x, grad_y, grad_z = gradient_gaussian(
                                        (xp, yp, zp), pos[ip_other], dist, filter_length
                                    )
                                    # volume of voronoi cell
                                    volume = (4.0 / 3.0) * cp.pi * hsml[ip_other] ** 3

                                    grad_x_Vec_x_Reg += (
                                        variable_x[ip_other] * grad_x * volume
                                    )
                                    grad_y_Vec_x_Reg += (
                                        variable_x[ip_other] * grad_y * volume
                                    )
                                    grad_z_Vec_x_Reg += (
                                        variable_x[ip_other] * grad_z * volume
                                    )

                                    grad_x_Vec_y_Reg += (
                                        variable_y[ip_other] * grad_x * volume
                                    )
                                    grad_y_Vec_y_Reg += (
                                        variable_y[ip_other] * grad_y * volume
                                    )
                                    grad_z_Vec_y_Reg += (
                                        variable_y[ip_other] * grad_z * volume
                                    )

                                    grad_x_Vec_z_Reg += (
                                        variable_z[ip_other] * grad_x * volume
                                    )
                                    grad_y_Vec_z_Reg += (
                                        variable_z[ip_other] * grad_y * volume
                                    )
                                    grad_z_Vec_z_Reg += (
                                        variable_z[ip_other] * grad_z * volume
                                    )

                                    numInteractingPartNew += 1

        # smoothVarRegister /= normalization

        grad_x_vec_x[oldIp] = grad_x_Vec_x_Reg
        grad_x_vec_y[oldIp] = grad_x_Vec_y_Reg
        grad_x_vec_z[oldIp] = grad_x_Vec_z_Reg
        grad_y_vec_x[oldIp] = grad_y_Vec_x_Reg
        grad_y_vec_y[oldIp] = grad_y_Vec_y_Reg
        grad_y_vec_z[oldIp] = grad_y_Vec_z_Reg
        grad_z_vec_x[oldIp] = grad_z_Vec_x_Reg
        grad_z_vec_y[oldIp] = grad_z_Vec_y_Reg
        grad_z_vec_z[oldIp] = grad_z_Vec_z_Reg

        hitsNeighbours[oldIp] = numInteractingPartNew
        # if not iterative we are going to be recycling the
        # hasConverged array to store how many particles it has searched
        # if (iterativeFilter == 0): # not iterative
        hasConverged[oldIp] = numSearchedPart
