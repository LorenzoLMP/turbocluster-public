import cupy as cp
from numba import cuda
import math

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




@cuda.jit
def find_tile_index_spherical(pos, radSpacing, rMin, rMax, tile_index,
                             phiSpacing, phiMin, theSpacing, theMin, type, power):
    """
    pos is relative to the origin.
    tile_index[i] = (R_i, phi_i, theta_i)
    radSpacing = (np.log10(_rMax) - np.log10(_rMin))/nSectRad
    rMin already includes the extra thickness (_rMin)
    We are assuming that phi can go from [0, 2\pi], and
    \theta from [0, \pi], but in theory on can modify to
    accept only a specific range.
    rMax include the extra thickness (_rMax)
    type = 0 is log
    type = 1 is power law with "power" exponent
    """
    ip = cuda.grid(1)

    if ip < pos.shape[0]:
        xp = pos[ip, 0]
        yp = pos[ip, 1]
        zp = pos[ip, 2]
        rp = math.sqrt(xp*xp + yp*yp + zp*zp)
        phi = math.atan2(yp, xp) % (2.0*math.pi)
        theta = math.acos(zp/rp)
        
        ip_tile_rad = find_tile_radial(rp, radSpacing, rMin, rMax, type, power)
        ip_tile_phi = int((phi - phiMin) // phiSpacing)
        ip_tile_the = int((theta - theMin) // theSpacing)
        # ip_tile_x = find_tile_single_dim(xp, npixs[0], tile_widths[0])
        # ip_tile_y = find_tile_single_dim(yp, npixs[1], tile_widths[1])
        # ip_tile_z = find_tile_single_dim(zp, npixs[2], tile_widths[2])
        tile_index[ip, 0] = ip_tile_rad
        tile_index[ip, 1] = ip_tile_phi
        tile_index[ip, 2] = ip_tile_the


@cuda.jit
def get_tile_information(tile_index, particles_per_tile, start_index_for_tile):
    """
    Need to do this on each block and then assemble
    """
    ip = cuda.grid(1)
    if ip < tile_index.shape[0]:
        ip_tile_x, ip_tile_y, ip_tile_z = tile_index[ip, :]
        cuda.atomic.add(particles_per_tile,
                        (ip_tile_x, ip_tile_y, ip_tile_z), 1)
        cuda.atomic.min(start_index_for_tile,
                        (ip_tile_x, ip_tile_y, ip_tile_z), ip)

# @cuda.jit
# def compactify_kernel(occupancy_arr, cumulative_occupancy_flat, npixs,
#                       Nmax, particles_per_tile,
#                       start_index_for_tile, numBlocksCompactGrid, 
#                       compactGrid):
#     """
#     """
#     ip = cuda.grid(1)
#     numBlocksFullGrid = int(npixs[0]*npixs[1]*npixs[2])
#     if (ip < numBlocksFullGrid):
#         newPos = int(cumulative_occupancy_flat[ip])
#         numBlocksNeeded = int(occupancy_arr[ip])
#         # we walk backwards like a shrimp
#         for i in range(numBlocksNeeded):
#             compactGrid[newPos - 1 - i, 0] = ip
#             compactGrid[newPos - 1 - i, 1] = start_index_for_tile[ip] + (numBlocksNeeded - 1 - i)*Nmax
#             if (i > 0): 
#                 compactGrid[newPos - 1 - i, 2] = Nmax
#             else: 
#                 compactGrid[newPos - 1 - i, 2] = particles_per_tile[ip] - \
#                                                     (numBlocksNeeded - 1) * Nmax
                                                        
            



class SphericalTiling:
    """
    """

    def __init__(self, positions, center, rMin, rMax, extra_layer_thickness,
                 nRadial=128, nPhi=128, nTheta=64, type='log', power=0,
                 threadsperblock=256):

        """
        extra_layer_thickness is for the whole selection
        maybe we need to differentiate between outer and inner radius?
        rMin, rMax are the domain computational boundaries
        chosen by the user
        _rMin, _rMax are the lower and upper limits of 
        the radial grid (computed by SphericalTiling)
        _rMin < rMin
        _rMax > rMax
        type = 0 is log
        type = 1 is power law with "power" exponent
        """

        Np = positions.shape[0]

        blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock

        # Copy positions
        self._pos = cp.array(positions)

        # self.tilebox_widths = widths + 2 * extra_layer_thickness

        nSectRad = nRadial
        nSectPhi = nPhi
        nSectThe = nTheta

        self.nSects = cp.array([nSectRad, nSectPhi, nSectThe])

        self.off_sets = center

        # need to take into account extra thickness
        _rMax = rMax + extra_layer_thickness
        if (rMin > extra_layer_thickness):
            _rMin = rMin - extra_layer_thickness
        else:
            # means that by including the extra thickness
            # _rMin becomes negative. 
            # rMin could either be exactly 0.0 or very small
            # Either case, we set 
            # _rMin = 1e-4 _rMax for the logarhithmic grid
            # Particles selected with the _do_region_selection_spherical()
            # may have radius < _rMin. In this case the find_tile_radial 
            # function will always put them in the 0-th radial sector
            if (type == 'log'):
                _rMin = _rMax * 1e-2
            elif (type == 'power-law'):
                _rMin = rMin * 0.0
        
        self._rMin = _rMin
        self._rMax = _rMax

        if (type == 'log'):
            # logarhithmic spacing (can also be a power-law...)
            # let's assume that the radial spacing is of the form
            # radialSpacing = np.logspace(np.log10(_rMin),np.log10(_rMax),nSectRad+1)
            self.radSpacing = (math.log10(_rMax) - math.log10(_rMin)) / nSectRad
            typeGrid = 0
        elif (type == 'power-law'):
            # let's assume that the radial spacing is uniform in the variable y
            # y = ( (r - _rMin)/ (_rMax - _rMin) ) ** power
            # so that y \in [0,1], which we split in nSectRad sectors
            self.radSpacing = 1.0 / nSectRad
            typeGrid = 1

        self.phiMin = 0.0
        self.theMin = 0.0
        self.phiSpacing = 2.0 * math.pi /  nPhi
        self.theSpacing = math.pi /  nTheta

        self.spacings = cp.array([self.radSpacing,
                                  self.phiSpacing,
                                  self.theSpacing])

        self._pos -= self.off_sets[None, :]

        # Get tile information
        # tile_index[i] = (R_i, phi_i, theta_i)
        self.tile_index = cp.zeros((Np, 3), dtype=int)

        self.particles_per_tile = cp.zeros((nSectRad, nSectPhi, nSectThe), dtype=int)
        # Initialize to value larger than the largest posssible tileindex
        self.start_index_for_tile = cp.ones(
            (nSectRad, nSectPhi, nSectThe), dtype=int) * Np + 1
                                
        find_tile_index_spherical[blocks_1d, threadsperblock](
            self._pos, self.radSpacing, _rMin, _rMax, self.tile_index,
            self.phiSpacing, self.phiMin, self.theSpacing, self.theMin, typeGrid, power)

        # here the indexing is such that:
        # idx = theta_k + nSectThe * ( phi_j + nSectPhi * R_i)
        self.sort_index = cp.argsort(self.tile_index[:, 0] * nSectThe * nSectPhi
                                     + self.tile_index[:, 1] * nSectThe +
                                     self.tile_index[:, 2])

        unsort_index = cp.zeros(Np, dtype=int)
        unsort_index[self.sort_index] = cp.arange(Np)
        self.unsort_index = unsort_index

        self.tile_index = self.tile_index[self.sort_index, :]

        get_tile_information[blocks_1d, threadsperblock](
            self.tile_index, self.particles_per_tile,
            self.start_index_for_tile)


    # def compactify_grid(self, Nmax):
    #     occupancy_arr = ((self.particles_per_tile + (Nmax - 1)) // Nmax).flatten()
    #     cumulative_occupancy = cp.cumsum(occupancy_arr)
    #     numBlocksCompactGrid = cumulative_occupancy[-1]
    #     compactGrid = cp.zeros((int(numBlocksCompactGrid),3),dtype=int)

    #     threadsperblock = 256
    #     numBlocksFullGrid = int(self.npixs[0]*self.npixs[1]*self.npixs[2])
    #     blocks_1d = (numBlocksFullGrid + (threadsperblock - 1)) // threadsperblock
    #     compactify_kernel[blocks_1d, threadsperblock](occupancy_arr,cumulative_occupancy, self.npixs, 
    #                                                  Nmax, self.particles_per_tile.flatten(), 
    #                                                  self.start_index_for_tile.flatten(), numBlocksCompactGrid, 
    #                                                  compactGrid)
    #     # compactGrid is an array of 3-tuples numBlocksCompactGrid long. 
    #     # for each block in numBlocksCompactGrid the 3-tuples is as follows
    #     # [id of the tile in the grid it refers to, id of first particle, num of particles it contains]

    #     self.compactGrid = compactGrid
        
