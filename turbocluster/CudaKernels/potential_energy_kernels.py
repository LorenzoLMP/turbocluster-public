import cupy as cp
from numba import cuda
import math
from .generic_kernels import distance


@cuda.jit(device=True, inline=True)
def grav_pot_kernel(r, h):

    # Eq.108 of 10.1111/j.1365-2966.2009.15715.x
    # with flipped sign
    # this is for all particle types
    phi = 1.0 / r
    u = r / h
    if u < 0.5:
        phi *= (
            (14.0 / 5.0) * u
            - (16.0 / 3.0) * u**3
            + (48.0 / 5.0) * u**5
            - (32.0 / 5.0) * u**6
        )
    elif u >= 0.5 and u < 1:
        phi *= 16.0 * u**4 - (48.0 / 5.0) * u**5 + (32.0 / 15.0) * u**6

    return phi


@cuda.jit()
def compute_potential_energy_coarse(
    pos,
    mass,
    smoothing_length,
    tile_index,
    start_index_for_tile,
    particles_per_tile,
    mass_per_tile,
    max_hsml_per_tile,
    tile_widths,
    tan_theta0,
    offsets,
    npixs,
    center,
    widths,
    potential_in_tile,
    tiles_hit,
):
    """
    this kernel is called with a thread assigned to each
    cartesian block of the grid
    """
    # threadindex
    ip = cuda.grid(1)
    numBlocks = int(npixs[0] * npixs[1] * npixs[2])
    if ip < numBlocks:

        # find tile indices
        ip_tile_z = ip % int(npixs[2])
        ip_tmp = int((ip - ip_tile_z) / int(npixs[2]))
        ip_tile_y = ip_tmp % int(npixs[1])
        ip_tile_x = ip_tmp // int(npixs[1])

        # coordinates of center of tile
        tile_coord_x = offsets[0] + (ip_tile_x + 0.5) * tile_widths[0]
        tile_coord_y = offsets[1] + (ip_tile_y + 0.5) * tile_widths[1]
        tile_coord_z = offsets[2] + (ip_tile_z + 0.5) * tile_widths[2]

        mass_this_tile = mass_per_tile[ip_tile_x, ip_tile_y, ip_tile_z]

        max_hsml_this_tile = max_hsml_per_tile[ip_tile_x, ip_tile_y, ip_tile_z]

        # H is the maximum transverse dimension of the tile
        H = math.sqrt(tile_widths[0] ** 2 + tile_widths[1] ** 2 + tile_widths[2] ** 2)

        for ip_other in range(0, numBlocks):

            # find other tile indices
            ip_other_tile_z = ip_other % int(npixs[2])
            ip_other_tmp = int((ip_other - ip_other_tile_z) / int(npixs[2]))
            ip_other_tile_y = ip_other_tmp % int(npixs[1])
            ip_other_tile_x = ip_other_tmp // int(npixs[1])

            # coordinates of center of other tile
            other_tile_coord_x = offsets[0] + (ip_other_tile_x + 0.5) * tile_widths[0]
            other_tile_coord_y = offsets[1] + (ip_other_tile_y + 0.5) * tile_widths[1]
            other_tile_coord_z = offsets[2] + (ip_other_tile_z + 0.5) * tile_widths[2]

            mass_other_tile = mass_per_tile[
                ip_other_tile_x, ip_other_tile_y, ip_other_tile_z
            ]
            max_hsml_other_tile = max_hsml_per_tile[
                ip_other_tile_x, ip_other_tile_y, ip_other_tile_z
            ]

            distance_ip_ip_other = distance(
                (tile_coord_x, tile_coord_y, tile_coord_z),
                (other_tile_coord_x, other_tile_coord_y, other_tile_coord_z),
            )

            max_hsml_tiles = max(max_hsml_this_tile, max_hsml_other_tile)
            # if (ip != ip_other):
            # potential_in_tile[ip_tile_x, ip_tile_y, ip_tile_z] -=
            # mass_this_tile * mass_other_tile / distance_ip_ip_other
            if distance_ip_ip_other < H / (2 * tan_theta0) or max_hsml_tiles > abs(
                distance_ip_ip_other - H
            ):
                # includes the case when ip == ip_other
                # do the particle-particle loop
                # the second condition refers to case when hsml is potentially
                # very large and partially overlaps with other tile
                pass

            else:
                # add to the potential the contribution of the
                # entire ip_other block
                # assume that the tiles/particles are far enough away
                # that it is not necessary to smoothen the potential
                potential_in_tile[ip_tile_x, ip_tile_y, ip_tile_z] -= (
                    mass_this_tile * mass_other_tile / distance_ip_ip_other
                )
                tiles_hit[ip_tile_x, ip_tile_y, ip_tile_z] += 1


@cuda.jit()
def compute_potential_energy_N2(
    pos,
    mass,
    smoothing_length,
    tile_index,
    start_index_for_tile,
    particles_per_tile,
    mass_per_tile,
    max_hsml_per_tile,
    tile_widths,
    tan_theta0,
    offsets,
    npixs,
    center,
    widths,
    potential_in_tile,
    particles_hit,
):
    """
    this kernel is called with a thread assigned to each
    particle
    """
    # threadindex
    ip = cuda.grid(1)
    numParticles = pos.shape[0]
    if ip < numParticles:

        # find tile indices
        ip_tile_x = tile_index[ip, 0]
        ip_tile_y = tile_index[ip, 1]
        ip_tile_z = tile_index[ip, 2]

        # coordinates of center of tile
        tile_coord_x = offsets[0] + (ip_tile_x + 0.5) * tile_widths[0]
        tile_coord_y = offsets[1] + (ip_tile_y + 0.5) * tile_widths[1]
        tile_coord_z = offsets[2] + (ip_tile_z + 0.5) * tile_widths[2]

        max_hsml_this_tile = max_hsml_per_tile[ip_tile_x, ip_tile_y, ip_tile_z]

        # H is the maximum transverse dimension of the tile
        H = math.sqrt(tile_widths[0] ** 2 + tile_widths[1] ** 2 + tile_widths[2] ** 2)
        dist_max = H / (2 * tan_theta0)

        ####################################
        # code to check what are the tiles that can overlap
        # with the tile in which particle ip resides
        ####################################

        ip_tile_x_min = ip_tile_x - (dist_max + tile_widths[0] / 2) // tile_widths[0]
        ip_tile_x_max = ip_tile_x + (dist_max + tile_widths[0] / 2) // tile_widths[0]

        ip_tile_y_min = ip_tile_y - (dist_max + tile_widths[1] / 2) // tile_widths[1]
        ip_tile_y_max = ip_tile_y + (dist_max + tile_widths[1] / 2) // tile_widths[1]

        ip_tile_z_min = ip_tile_z - (dist_max + tile_widths[2] / 2) // tile_widths[2]
        ip_tile_z_max = ip_tile_z + (dist_max + tile_widths[2] / 2) // tile_widths[2]

        ip_tile_x_min = max(ip_tile_x_min, 0)
        ip_tile_y_min = max(ip_tile_y_min, 0)
        ip_tile_z_min = max(ip_tile_z_min, 0)

        ip_tile_x_max = min(ip_tile_x_max, npixs[0] - 1)
        ip_tile_y_max = min(ip_tile_y_max, npixs[1] - 1)
        ip_tile_z_max = min(ip_tile_z_max, npixs[2] - 1)

        ####################################
        # end of code to check overlap
        ####################################

        for ip_other_tile_x in range(ip_tile_x_min, ip_tile_x_max + 1):
            for ip_other_tile_y in range(ip_tile_y_min, ip_tile_y_max + 1):
                for ip_other_tile_z in range(ip_tile_z_min, ip_tile_z_max + 1):
                    # coordinates of center of other tile
                    other_tile_coord_x = (
                        offsets[0] + (ip_other_tile_x + 0.5) * tile_widths[0]
                    )
                    other_tile_coord_y = (
                        offsets[1] + (ip_other_tile_y + 0.5) * tile_widths[1]
                    )
                    other_tile_coord_z = (
                        offsets[2] + (ip_other_tile_z + 0.5) * tile_widths[2]
                    )

                    distance_tile_other_tile = distance(
                        (tile_coord_x, tile_coord_y, tile_coord_z),
                        (other_tile_coord_x, other_tile_coord_y, other_tile_coord_z),
                    )

                    start_index_other_tile = start_index_for_tile[
                        ip_other_tile_x, ip_other_tile_y, ip_other_tile_z
                    ]
                    num_particles_other_tile = particles_per_tile[
                        ip_other_tile_x, ip_other_tile_y, ip_other_tile_z
                    ]
                    max_hsml_other_tile = max_hsml_per_tile[
                        ip_other_tile_x, ip_other_tile_y, ip_other_tile_z
                    ]

                    max_hsml_tiles = max(max_hsml_this_tile, max_hsml_other_tile)

                    if distance_tile_other_tile < H / (
                        2 * tan_theta0
                    ) or max_hsml_tiles > abs(distance_tile_other_tile - H):
                        # includes the case when ip == ip_other
                        # do the particle-particle loop
                        # the second condition refers to case when hsml is potentially
                        # very large and partially overlaps with other tile
                        # ip, ip_other here refer to particle indices
                        for ip_other in range(
                            start_index_other_tile,
                            start_index_other_tile + num_particles_other_tile,
                        ):
                            # for symmetry use the max hsml of the two particles in
                            # calculation of potential
                            hsml = max(smoothing_length[ip], smoothing_length[ip_other])
                            # exclude self-interaction
                            if ip != ip_other:
                                # pass
                                r = distance(pos[ip], pos[ip_other])
                                pot_tmp = (
                                    -mass[ip]
                                    * mass[ip_other]
                                    * grav_pot_kernel(r, hsml)
                                )
                            else:
                                # pass
                                # Eq.108 of 10.1111/j.1365-2966.2009.15715.x
                                # limit for r->0
                                pot_tmp = (
                                    -mass[ip] * mass[ip_other] * (14.0 / 5.0) / hsml
                                )

                            cuda.atomic.add(
                                potential_in_tile,
                                (ip_tile_x, ip_tile_y, ip_tile_z),
                                pot_tmp,
                            )
                            particles_hit[ip] += 1
