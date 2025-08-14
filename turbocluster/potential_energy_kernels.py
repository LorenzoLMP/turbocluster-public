import cupy as cp
from numba import cuda
from .generic_kernels import *

@cuda.jit()
def compute_potential_energy(pos, mass, smoothing_length, 
                             tile_index, start_index_for_tile,
                             particles_per_tile, mass_per_tile,
                             tile_widths, tan_theta0,
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
        tile_coord_x = offsets[0] + (ip_tile_x + 0.5) * tile_widths[0] 
        tile_coord_y = offsets[1] + (ip_tile_y + 0.5) * tile_widths[1] 
        tile_coord_z = offsets[2] + (ip_tile_z + 0.5) * tile_widths[2] 

        # H is the maximum transverse dimension of the tile 
        H = math.sqrt(tile_widths[0]**2 + tile_widths[1]**2 + tile_widths[2]**2)
        
        # start_index_for_tile[ip], particles_per_tile[ip]

        for ip_other in range(0,numBlocks):

            # find other tile indices
            ip_other_tile_z = ip_other % int(npixs[2])
            ip_other_tmp = int((ip_other - ip_other_tile_z)/int(npixs[2]))
            ip_other_tile_y = ip_other_tmp % int(npixs[1])
            ip_other_tile_x = ip_other_tmp // int(npixs[1])
        
            # coordinates of center of other tile
            other_tile_coord_x = offsets[0] + (ip_other_tile_x + 0.5) * tile_widths[0] 
            other_tile_coord_y = offsets[1] + (ip_other_tile_y + 0.5) * tile_widths[1] 
            other_tile_coord_z = offsets[2] + (ip_other_tile_z + 0.5) * tile_widths[2] 

            
            distance_ip_ip_other = distance((tile_coord_x, tile_coord_y, tile_coord_z), 
                                            (other_tile_coord_x, other_tile_coord_y, other_tile_coord_z))

            if distance_ip_ip_other < H/(2*tan_theta0):
                ## includes the case when ip == ip_other
                ## do the particle-particle loop
                ## making sure not to include self-interaction
                ## i, j here refer to particle indices
                for i in range(start_index_for_tile[ip], 
                                start_index_for_tile[ip] + particles_per_tile[ip]):
                    for j in range(start_index_for_tile[ip_other], 
                                    start_index_for_tile[ip_other] + particles_per_tile[ip_other]):
                        ## exclude self-interaction
                        if (i != j):
                            ## TODO: add smoothing length here in the denominator
                            potential_in_tile[ip] -= mass[i]*mass[j]/distance(pos[i], pos[j])
            else:
                ## add to the potential the contribution of the 
                ## entire ip_other block
                potential_in_tile[ip] -= mass_per_tile[ip]*mass_per_tile[ip_other]/distance_ip_ip_other

        


        