import cupy as cp
from numba import cuda
import math


@cuda.jit(device=True, inline=True)
def weight_cic(xg, xi, h, Deltax):
    """
    xg is the coordinate of the grid point
    xi is the coordinate of the particle
    h is the support of the particle (the hsml diameter)
    Deltax is the spacing between gridpoints
    """
    t = Deltax / 2.0 + h / 2.0 - math.fabs(xg - xi)
    minOverlap = min(Deltax, h)
    if t < 0.0:
        return 0.0  # this is ok
    elif t >= 0.0 and t < minOverlap:
        return t / h
    elif t >= minOverlap:
        return minOverlap / h


@cuda.jit(device=True, inline=True)
def weight_tsc(xg, xi, h, Deltax):
    """
    xg is the coordinate of the grid point
    xi is the coordinate of the particle
    h is the the hsml diameter (the support of
    the particle is twice that)
    Deltax is the spacing between gridpoints
    """
    t = Deltax / 2.0 + h - math.fabs(xg - xi)
    weight = 0.0
    if 2.0 * h < Deltax:
        # the particle support is smaller than the
        # cartesian cell
        if t < 0.0:
            weight = 0.0
        elif t > 0.0 and t <= h:
            weight = t * t / (2.0 * h)
        elif t > h and t <= 2.0 * h:
            weight = h - ((2.0 * h - t) ** 2) / (2.0 * h)
        elif t > 2.0 * h:
            weight = h
    else:
        # the particle support is bigger than the
        # cartesian cell (Delta x < 2.*h)
        if t < 0.0:
            weight = 0.0
        elif t > 0.0 and t <= Deltax:
            weight = t * t / (2.0 * h)
        elif t > Deltax:
            if t <= h:
                weight = (1.0 / h) * Deltax * (t - Deltax / 2.0)
            elif t > h:
                weight = (1.0 / h) * (Deltax * (t - Deltax / 2.0) - (t - h) ** 2)

    # we normalize weight such that the total area is 1
    weight /= h

    return weight


@cuda.jit(device=True, inline=True)
def weight_psc(xg, xi, h, Deltax):
    """
    xg is the coordinate of the grid point
    xi is the coordinate of the particle
    h is the the hsml diameter (the support of
    the particle is 3 times that)
    Deltax is the spacing between gridpoints
    """
    # t is the overlap
    t = Deltax / 2.0 + 3.0 * h / 2.0 - math.fabs(xg - xi)
    weight = 0.0

    if t <= 0.0:
        weight = 0.0
    elif t > 0.0 and t <= h:
        if t <= Deltax:
            weight = t**3 / (6.0 * h**3)
        else:
            weight = Deltax / (6.0 * h**3) * (Deltax**2 - 3.0 * Deltax * t + 3 * t**2)
    elif t > h and t <= 2.0 * h:
        if t <= Deltax:
            weight = (
                3.0 * h**3 - 9.0 * h**2 * t + 9.0 * h * t**2 - 2.0 * t**3
            ) / (6.0 * h**3)
        elif t > Deltax and t <= Deltax + h:
            weight = (h - t) * (2.0 * h**2 - 7.0 * h * t + 2.0 * t**2) / (6.0 * h**3)
            weight += (h**3 - (t - Deltax) ** 3) / (6.0 * h**3)
        else:
            weight = Deltax * (
                -2.0 * Deltax**2
                - 9.0 * Deltax * h
                - 9.0 * h**2
                + 6.0 * (Deltax + 3.0 * h) * t
                - 6.0 * t**2
            )
            weight /= 6.0 * h**3

    elif t > 2.0 * h and t <= 3.0 * h:
        if t <= Deltax:
            weight = (t**3 - 9.0 * h * t**2 + 27.0 * h**2 * t - 21.0 * h**3) / (
                6.0 * h**3
            )
        else:
            weight = (
                Deltax**3
                - 21.0 * h**3
                - 3.0 * (Deltax**2 - 9.0 * h**2) * t
                + 3.0 * (Deltax - 3.0 * h) * t**2
            )
            weight /= 6.0 * h**3

    else:  # t > 3h
        weight = 1.0

    return weight


@cuda.jit()
def deposit_on_grid(
    pos,
    hsml,
    tile_widths,
    variable,
    weights,
    offsets,
    npixs,
    center,
    widths,
    deposited_var,
    scratch,
    kernel_type,
):
    """
    scratch has same dimensions as deposited_var (npixs[0], npixs[1], npixs[2])
    xgrid has shape (npixs[0],)
    ygrid has shape (npixs[1],)
    zgrid has shape (npixs[2],)
    """
    # threadindex
    ip = cuda.grid(1)

    # particle position
    xp = pos[ip, 0]
    yp = pos[ip, 1]
    zp = pos[ip, 2]

    # the particle diameter is hsml
    # however, based on the order of the
    # interpolation n, the *support* (size) of the particle is
    # n \times hsml
    # for NGP: support = delta function
    # for CIC: support = 1 x hsml
    # for TSC: support = 2 x hsml
    # for PCS: support = 3 x hsml
    suppPart = kernel_type * hsml[ip]

    # NOTE: in the kernel call use hsml and
    # NOT suppPart

    # for mass-weighting weights will be the
    # masses of the particles
    weightVar = weights[ip]

    # the 0.5 * suppPart is because we are considering
    # the left half and right half
    xmin = center[0] - widths[0] / 2 - 0.5 * suppPart
    xmax = center[0] + widths[0] / 2 + 0.5 * suppPart

    ymin = center[1] - widths[1] / 2 - 0.5 * suppPart
    ymax = center[1] + widths[1] / 2 + 0.5 * suppPart

    zmin = center[2] - widths[2] / 2 - 0.5 * suppPart
    zmax = center[2] + widths[2] / 2 + 0.5 * suppPart

    # xmin = center[0] - widths[0] / 2 - 0.0 * suppPart
    # xmax = center[0] + widths[0] / 2 + 0.0 * suppPart

    # ymin = center[1] - widths[1] / 2 - 0.0 * suppPart
    # ymax = center[1] + widths[1] / 2 + 0.0 * suppPart

    # zmin = center[2] - widths[2] / 2 - 0.0 * suppPart
    # zmax = center[2] + widths[2] / 2 + 0.0 * suppPart

    sidelength_x, sidelength_y, sidelength_z = widths
    nx, ny, nz = npixs
    voxel = tile_widths[0] * tile_widths[1] * tile_widths[2]

    # Check if this cell/particle is inside domain
    inside_domain = False
    if (xp > xmin) and (xp < xmax):
        if (yp > ymin) and (yp < ymax):
            if (zp > zmin) and (zp < zmax):
                inside_domain = True

    if inside_domain:

        # can be negative, i.e. the particle is outside interpolating region
        # but contributes to the deposition
        ip_tile_x = int((xp - offsets[0]) // tile_widths[0])
        ip_tile_y = int((yp - offsets[1]) // tile_widths[1])
        ip_tile_z = int((zp - offsets[2]) // tile_widths[2])

        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0]
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1]
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2]

        weight_tmp = 0.0

        ip_tile_x_min = (
            ip_tile_x
            - (-delta_x + 0.5 * suppPart + tile_widths[0] / 2) // tile_widths[0]
        )
        ip_tile_x_max = (
            ip_tile_x
            + (delta_x + 0.5 * suppPart + tile_widths[0] / 2) // tile_widths[0]
        )

        ip_tile_y_min = (
            ip_tile_y
            - (-delta_y + 0.5 * suppPart + tile_widths[1] / 2) // tile_widths[1]
        )
        ip_tile_y_max = (
            ip_tile_y
            + (delta_y + 0.5 * suppPart + tile_widths[1] / 2) // tile_widths[1]
        )

        ip_tile_z_min = (
            ip_tile_z
            - (-delta_z + 0.5 * suppPart + tile_widths[2] / 2) // tile_widths[2]
        )
        ip_tile_z_max = (
            ip_tile_z
            + (delta_z + 0.5 * suppPart + tile_widths[2] / 2) // tile_widths[2]
        )

        if ip_tile_x_min < 0:
            ip_tile_x_min = 0
        if ip_tile_y_min < 0:
            ip_tile_y_min = 0
        if ip_tile_y_min < 0:
            ip_tile_y_min = 0
        if ip_tile_x_max > nx - 1:
            ip_tile_x_max = nx - 1
        if ip_tile_y_max > ny - 1:
            ip_tile_y_max = ny - 1
        if ip_tile_y_max > nz - 1:
            ip_tile_y_max = nz - 1

        # kernel_type == 0 is nearest-grid-point (NGP)
        if kernel_type == 0:
            # these are the indices of the closest grid-point
            # if particle is in tile i, its xcoord is \in [x_i, x_(i+1))
            # if delta_x > 0 it means that particle is closer to
            # x_(i+1) than to x_i
            xgrid_idx = ip_tile_x
            if delta_x > 0:
                xgrid_idx += 1
            ygrid_idx = ip_tile_y
            if delta_y > 0:
                ygrid_idx += 1
            zgrid_idx = ip_tile_z
            if delta_z > 0:
                zgrid_idx += 1

            # there could be out-of-bound problems
            # need to select only gridpoints that belong to the grid
            if (xgrid_idx >= 0) and (xgrid_idx < nx + 1):
                if (ygrid_idx >= 0) and (ygrid_idx < ny + 1):
                    if (zgrid_idx >= 0) and (zgrid_idx < nz + 1):

                        weight_tmp = weightVar / voxel
                        # weight_tmp = 1.0
                        # the following two need to be atomic add
                        cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx), 1.0)
                        cuda.atomic.add(
                            deposited_var,
                            (xgrid_idx, ygrid_idx, zgrid_idx),
                            variable[ip] * weight_tmp,
                        )

        # kernel_type == 1 is cloud-in-cell (CIC), particle is a cube
        elif kernel_type == 1:
            cuda.atomic.add(scratch, (0, 0, 0), 1.0)
            # these are the indices of the potentially overlapping grid-points
            # if particle is in tile i, its xcoord is \in [x_i, x_(i+1))
            # if with its hslm it overlaps tiles from ip_tile_x_min to
            # ip_tile_x_max (inclusive), the grid indices that it can overlap with
            # go from ip_tile_x_min  to ip_tile_x_max + 1 (inclusive)
            # this is because every gridpoint x(i),y(j) "owns"
            # a surrounding area going from [x(i)-Deltax/2, x(i)+Deltax/2]
            # and similarly for y
            # e.g. if particle in tile 2 overlaps with tiles 1-3
            # the grid points it could potentially overlap with
            # go from x(1) to x(4) (inclusive)
            # y(4)   |_______|_______|_______|_______|_
            #        |       |       |       |       |
            #        |    ~~~|~~~  - | - - - |- -    |
            # y(3)   |___'___|___"___|_______|___'___|_
            #        |   '   |   "   |       |   '   |
            #        |   '~~~|~~~"   |   x   |   '   |
            # y(2)   |_______|___'___|_______|___'___|_
            #        |       |   '   |       |   '   |
            #        |       |    - -| - - - |- -    |
            # y(1)   |_______|_______|_______|_______|_
            #        |       |       | tile  |       |
            #        |       |       |   2   |       |
            #       x(0)    x(1)    x(2)    x(3)    x(4)

            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 2):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 2):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 2):

                        # indices of the grid point
                        xgrid_idx = tile_x
                        ygrid_idx = tile_y
                        zgrid_idx = tile_z

                        # coords of the grid point
                        xcoord = center[0] - widths[0] / 2 + tile_x * tile_widths[0]
                        ycoord = center[1] - widths[1] / 2 + tile_y * tile_widths[1]
                        zcoord = center[2] - widths[2] / 2 + tile_z * tile_widths[2]

                        # there could be out-of-bound problems
                        # need to select only tiles that belong to the grid
                        if (xgrid_idx >= 0) and (xgrid_idx < nx + 1):
                            if (ygrid_idx >= 0) and (ygrid_idx < ny + 1):
                                if (zgrid_idx >= 0) and (zgrid_idx < nz + 1):

                                    # write CIC deposition routine
                                    # weight in x dimension
                                    weight_tmp = (
                                        weight_cic(xcoord, xp, hsml[ip], tile_widths[0])
                                        * weight_cic(
                                            ycoord, yp, hsml[ip], tile_widths[1]
                                        )
                                        * weight_cic(
                                            zcoord, zp, hsml[ip], tile_widths[2]
                                        )
                                        * weightVar
                                        / voxel
                                    )

                                    # # the following two need to be atomic add
                                    # cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx),
                                    #                 1.0)
                                    cuda.atomic.add(
                                        deposited_var,
                                        (xgrid_idx, ygrid_idx, zgrid_idx),
                                        variable[ip] * weight_tmp,
                                    )

        # kernel_type == 2 is Triangular Shape Cloud (TSC), particle is sort of pyramid
        elif kernel_type == 2:
            cuda.atomic.add(scratch, (0, 0, 0), 1.0)
            # the overlapping picture is the same as before, only the support
            # has now increased
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 2):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 2):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 2):

                        # indices of the grid point
                        xgrid_idx = tile_x
                        ygrid_idx = tile_y
                        zgrid_idx = tile_z

                        # coords of the grid point
                        xcoord = center[0] - widths[0] / 2 + tile_x * tile_widths[0]
                        ycoord = center[1] - widths[1] / 2 + tile_y * tile_widths[1]
                        zcoord = center[2] - widths[2] / 2 + tile_z * tile_widths[2]

                        # there could be out-of-bound problems
                        # need to select only tiles that belong to the grid
                        if (xgrid_idx >= 0) and (xgrid_idx < nx + 1):
                            if (ygrid_idx >= 0) and (ygrid_idx < ny + 1):
                                if (zgrid_idx >= 0) and (zgrid_idx < nz + 1):

                                    # TSC deposition routine
                                    weight_tmp = (
                                        weight_tsc(xcoord, xp, hsml[ip], tile_widths[0])
                                        * weight_tsc(
                                            ycoord, yp, hsml[ip], tile_widths[1]
                                        )
                                        * weight_tsc(
                                            zcoord, zp, hsml[ip], tile_widths[2]
                                        )
                                        * weightVar
                                        / voxel
                                    )
                                    # weight_tmp = 1.0
                                    # the following two need to be atomic add
                                    # cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx),
                                    #                 1.0)
                                    cuda.atomic.add(
                                        deposited_var,
                                        (xgrid_idx, ygrid_idx, zgrid_idx),
                                        variable[ip] * weight_tmp,
                                    )

        # kernel_type == 3 is Piecewise Cubic Splina (PCS), particle is sort of
        # 2nd order polynomial
        elif kernel_type == 3:
            cuda.atomic.add(scratch, (0, 0, 0), 1.0)
            # the overlapping picture is the same as before, only the support
            # has now increased
            for tile_x in range(ip_tile_x_min, ip_tile_x_max + 2):
                for tile_y in range(ip_tile_y_min, ip_tile_y_max + 2):
                    for tile_z in range(ip_tile_z_min, ip_tile_z_max + 2):

                        # indices of the grid point
                        xgrid_idx = tile_x
                        ygrid_idx = tile_y
                        zgrid_idx = tile_z

                        # coords of the grid point
                        xcoord = center[0] - widths[0] / 2 + tile_x * tile_widths[0]
                        ycoord = center[1] - widths[1] / 2 + tile_y * tile_widths[1]
                        zcoord = center[2] - widths[2] / 2 + tile_z * tile_widths[2]

                        # there could be out-of-bound problems
                        # need to select only tiles that belong to the grid
                        if (xgrid_idx >= 0) and (xgrid_idx < nx + 1):
                            if (ygrid_idx >= 0) and (ygrid_idx < ny + 1):
                                if (zgrid_idx >= 0) and (zgrid_idx < nz + 1):

                                    # PCS deposition routine
                                    weight_tmp = (
                                        weight_tsc(xcoord, xp, hsml[ip], tile_widths[0])
                                        * weight_tsc(
                                            ycoord, yp, hsml[ip], tile_widths[1]
                                        )
                                        * weight_tsc(
                                            zcoord, zp, hsml[ip], tile_widths[2]
                                        )
                                        * weightVar
                                        / voxel
                                    )
                                    # weight_tmp = 1.0
                                    # the following two need to be atomic add
                                    # cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx),
                                    #                 1.0)
                                    cuda.atomic.add(
                                        deposited_var,
                                        (xgrid_idx, ygrid_idx, zgrid_idx),
                                        variable[ip] * weight_tmp,
                                    )
