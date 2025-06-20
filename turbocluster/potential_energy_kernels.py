import cupy as cp
from numba import cuda
from .generic_kernels import *

@cuda.jit()
def compute_potential_energy(pos, mass, smoothing_length, 
                             tile_index, start_index_for_tile,
                             particles_per_tile, tile_widths,
                             offsets, npixs, center, widths, 
                             potential_in_tile):

    """
    this kernel is called with a thread assigned to each 
    cartesian block of the grid
    """
    # threadindex
    ip = cuda.grid(1)
    numBlocks = int(npixs[0]*npixs[1]*npixs[2])
    if (ip < numBlocks):


        # find tile indices
        ip_tile_z = ip % int(npixs[2])
        ip_tmp = int((ip - ip_tile_z)/int(npixs[2]))
        ip_tile_y = ip_tmp % int(npixs[1])
        ip_tile_x = ip_tmp // int(npixs[1])
    
        # coordinates of center of tile
        tile_coord_x = offsets[0] - (ip_tile_x + 0.5) * tile_widths[0] 
        tile_coord_y = offsets[1] - (ip_tile_y + 0.5) * tile_widths[1] 
        tile_coord_z = offsets[2] - (ip_tile_z + 0.5) * tile_widths[2] 

        # start_index_for_tile[ip], particles_per_tile[ip]

        


        