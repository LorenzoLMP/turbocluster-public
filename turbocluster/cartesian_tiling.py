import cupy as cp
from numba import cuda


@cuda.jit(device=True, inline=True)
def find_tile_single_dim(particle_pos, nx, tile_width):
    """
    """
    tile_index = int(particle_pos / tile_width)
    return tile_index


@cuda.jit
def find_tile_index(pos, npixs, tile_widths, tile_index):
    """
    Assuming equal width for simplicity for now
    """
    ip = cuda.grid(1)

    if ip < pos.shape[0]:
        xp = pos[ip, 0]
        yp = pos[ip, 1]
        zp = pos[ip, 2]
        ip_tile_x = find_tile_single_dim(xp, npixs[0], tile_widths[0])
        ip_tile_y = find_tile_single_dim(yp, npixs[1], tile_widths[1])
        ip_tile_z = find_tile_single_dim(zp, npixs[2], tile_widths[2])
        tile_index[ip, 0] = ip_tile_x
        tile_index[ip, 1] = ip_tile_y
        tile_index[ip, 2] = ip_tile_z


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

@cuda.jit
def compactify_kernel(occupancy_arr, cumulative_occupancy_flat, npixs,
                      Nmax, particles_per_tile,
                      start_index_for_tile, numBlocksCompactGrid, 
                      compactGrid):
    """
    """
    ip = cuda.grid(1)
    numBlocksFullGrid = int(npixs[0]*npixs[1]*npixs[2])
    if (ip < numBlocksFullGrid):
        newPos = int(cumulative_occupancy_flat[ip])
        numBlocksNeeded = int(occupancy_arr[ip])
        # we walk backwards like a shrimp
        for i in range(numBlocksNeeded):
            compactGrid[newPos - 1 - i, 0] = ip
            compactGrid[newPos - 1 - i, 1] = start_index_for_tile[ip] + (numBlocksNeeded - 1 - i)*Nmax
            if (i > 0): 
                compactGrid[newPos - 1 - i, 2] = Nmax
            else: 
                compactGrid[newPos - 1 - i, 2] = particles_per_tile[ip] - \
                                                    (numBlocksNeeded - 1) * Nmax
                                                        
            



class CartesianTiling:
    """
    """

    def __init__(self, positions, center, widths, extra_layer_thickness,
                 npix=128, threadsperblock=256):

        Np = positions.shape[0]

        blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock

        # Copy positions
        self._pos = cp.array(positions)

        self.tilebox_widths = widths + 2 * extra_layer_thickness

        npix_x = npix
        npix_y = int(self.tilebox_widths[1] / self.tilebox_widths[0] * npix)
        npix_z = int(self.tilebox_widths[2] / self.tilebox_widths[0] * npix)

        self.npixs = cp.array([npix_x, npix_y, npix_z])

        self.off_sets = center - self.tilebox_widths / 2.0

        self.tile_widths = self.tilebox_widths / self.npixs

        self._pos -= self.off_sets[None, :]

        # Get tile information
        self.tile_index = cp.zeros((Np, 3), dtype=int)

        self.particles_per_tile = cp.zeros((npix_x, npix_y, npix_z), dtype=int)
        # Initialize to value larger than the largest posssible tileindex
        self.start_index_for_tile = cp.ones(
            (npix_x, npix_y, npix_z), dtype=int) * Np + 1

        find_tile_index[blocks_1d, threadsperblock](
            self._pos, self.npixs, self.tile_widths, self.tile_index)

        self.sort_index = cp.argsort(self.tile_index[:, 2] * npix_x * npix_y
                                     + self.tile_index[:, 1] * npix_x +
                                     self.tile_index[:, 0])

        unsort_index = cp.zeros(Np, dtype=int)
        unsort_index[self.sort_index] = cp.arange(Np)
        self.unsort_index = unsort_index

        self.tile_index = self.tile_index[self.sort_index, :]

        get_tile_information[blocks_1d, threadsperblock](
            self.tile_index, self.particles_per_tile,
            self.start_index_for_tile)


    def compactify_grid(self, Nmax):
        occupancy_arr = ((self.particles_per_tile + (Nmax - 1)) // Nmax).flatten()
        cumulative_occupancy = cp.cumsum(occupancy_arr)
        numBlocksCompactGrid = cumulative_occupancy[-1]
        compactGrid = cp.zeros((int(numBlocksCompactGrid),3),dtype=int)

        threadsperblock = 256
        numBlocksFullGrid = int(self.npixs[0]*self.npixs[1]*self.npixs[2])
        blocks_1d = (numBlocksFullGrid + (threadsperblock - 1)) // threadsperblock
        compactify_kernel[blocks_1d, threadsperblock](occupancy_arr,cumulative_occupancy, self.npixs, 
                                                     Nmax, self.particles_per_tile.flatten(), 
                                                     self.start_index_for_tile.flatten(), numBlocksCompactGrid, 
                                                     compactGrid)
        # compactGrid is an array of 3-tuples numBlocksCompactGrid long. 
        # for each block in numBlocksCompactGrid the 3-tuples is as follows
        # [id of the tile in the grid it refers to, id of first particle, num of particles it contains]

        self.compactGrid = compactGrid
        
