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


class CartesianTiling:
    """
    """

    def __init__(self, positions, npix=128, threadsperblock=256):

        Np = positions.shape[0]

        blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock

        # Copy positions
        self._pos = cp.array(positions)

        # Convert self._position to units from [0, 1) * 2**L
        self.off_sets = cp.min(self._pos, axis=0)
        self._pos -= self.off_sets[None, :]

        max_pos = cp.max(self._pos, axis=0)

        self.conversion_factor = (npix - 1) / max_pos[0]
        self.npixs = (max_pos * self.conversion_factor).astype(int) + 1

        npix_x = int(self.npixs[0])
        npix_y = int(self.npixs[1])
        npix_z = int(self.npixs[2])

        self.tile_widths = max_pos / self.npixs
        self.tile_widths_internal = self.tile_widths * self.conversion_factor

        self._pos *= self.conversion_factor

        # Get tile information
        self.tile_index = cp.zeros((Np, 3), dtype=int)

        self.particles_per_tile = cp.zeros((npix_x, npix_y, npix_z), dtype=int)
        # Initialize to value larger than the largest posssible tileindex
        self.start_index_for_tile = cp.ones(
            (npix_x, npix_y, npix_z), dtype=int) * Np + 1

        find_tile_index[blocks_1d, threadsperblock](
            self._pos, self.npixs, self.tile_widths_internal, self.tile_index)

        self.sort_index = cp.argsort(self.tile_index[:, 2] * npix_x * npix_y
                                     + self.tile_index[:, 1] * npix_x +
                                     self.tile_index[:, 0])

        self.tile_index = self.tile_index[self.sort_index, :]

        get_tile_information[blocks_1d, threadsperblock](
            self.tile_index, self.particles_per_tile,
            self.start_index_for_tile)
