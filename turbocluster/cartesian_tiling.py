import numpy as np
import cupy as cp
from numba import cuda
import numba


@cuda.jit(device=True, inline=True)
def find_tile_single_dim(particle_pos, nx, tile_width):
    """
    Would be faster to find the half that it is...
    """
    tile_index = 0
    pos = tile_width
    while particle_pos > pos:
        pos += tile_width
        tile_index += 1
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
        cuda.atomic.add(particles_per_tile, (ip_tile_x, ip_tile_y, ip_tile_z), 1)
        cuda.atomic.min(start_index_for_tile, (ip_tile_x, ip_tile_y, ip_tile_z), ip)

class CartesianTiling:
    """
    """

    def __init__(self, pos, center, widths,
                 npix, threadsperblock):
        
        npix_x = npix
        npix_y = int(widths[1] / widths[0] * npix)
        npix_z = int(widths[2] / widths[0] * npix)



        # Get tile information
        particles_per_tile = cp.zeros((npix_x, npix_y, npix_z), dtype=int)
        # Initialize to value larger than the largest posssible tileindex
        start_index_for_tile = cp.ones((npix_x, npix_y, npix_z), dtype=int) * pos.shape[0] + 1
        get_tile_information(tile_index, particles_per_tile, start_index_for_tile)

        # self.sort_index = 
        # self.tile_index = 

    def find_tiles(self):
        pass