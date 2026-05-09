#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from numba import cuda
import paicos as pa
import numpy as np
import cupy as cp
import turbocluster as tc
import math
from numba import cuda

# A snapshot object
# snap = pa.Snapshot(pa.data_dir, 247)
snap = pa.Snapshot('/lustre/astro/berlok/zoom-simulations-new-ics/halo_0003/adiabatic-mhd/zoom4_ics_v1/output', 247)
# snap = pa.Snapshot('/lustre/astro/berlok/zoom-simulations-new-ics/halo_0003/tng/zoom12_ics_v1/output', 247)
center = snap.Cat.Group['GroupPos'][0]
widths = np.array([500., 500., 500.], dtype=float)

# m_filter = 1000*snap.mass
# filter_length = (np.cbrt(3*m_filter/(4*np.pi*snap['0_Density']))).arepo
filter_length = 2*snap['0_Diameters']


# In[ ]:


@cuda.jit(device=True, inline=True)
def weight_cic(xg, xi, h, Deltax):
    """
    xg is the coordinate of the grid point
    xi is the coordinate of the particle
    h is the sph radius of the particle
    Deltax is the spacing between gridpoints
    """
    t = Deltax/2.0 + h - math.fabs(xg-xi)
    minOverlap = min(Deltax,2.*h)
    if (t < 0.0):
        return 0.0
    elif (t >= 0.0 and t < minOverlap):
        return t/(2.*h)
    elif (t >= minOverlap):
        return minOverlap/(2.*h)


@cuda.jit(lineinfo=True)
def deposit_on_grid(pos, hsml, tile_widths,
                 variable, weights, offsets, npixs, center, widths, deposited_var, 
                 scratch, kernel_type):
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

    # particle radius (hsml)
    hsmlPart = hsml[ip]

    # for mass-weighting weights will be the 
    # masses of the particles
    weightVar = weights[ip]

    xmin = center[0] - widths[0] / 2 - hsmlPart
    xmax = center[0] + widths[0] / 2 + hsmlPart

    ymin = center[1] - widths[1] / 2 - hsmlPart
    ymax = center[1] + widths[1] / 2 + hsmlPart

    zmin = center[2] - widths[2] / 2 - hsmlPart
    zmax = center[2] + widths[2] / 2 + hsmlPart

    

    sidelength_x, sidelength_y, sidelength_z = widths
    nx, ny, nz = npixs

    # Check if this cell/particle is inside domain
    inside_domain = False
    if (xp > xmin) and (xp < xmax):
        if (yp > ymin) and (yp < ymax):
            if (zp > zmin) and (zp < zmax):
                inside_domain = True

    if inside_domain:

        # can be negative, i.e. the particle is outside interpolating region
        # but contributes to the deposition
        ip_tile_x = int( (xp - offsets[0]) // tile_widths[0])
        ip_tile_y = int( (yp - offsets[1]) // tile_widths[1])
        ip_tile_z = int( (zp - offsets[2]) // tile_widths[2])

        # relative coordinates w.r.t. center of tile
        delta_x = xp - offsets[0] - (ip_tile_x + 0.5) * tile_widths[0] 
        delta_y = yp - offsets[1] - (ip_tile_y + 0.5) * tile_widths[1] 
        delta_z = zp - offsets[2] - (ip_tile_z + 0.5) * tile_widths[2] 

        weight_tmp = 0.0

        ip_tile_x_min = ip_tile_x - (- delta_x + \
                        hsmlPart + tile_widths[0] / 2) // tile_widths[0] 
        ip_tile_x_max = ip_tile_x + (delta_x +   \
                        hsmlPart + tile_widths[0] / 2) // tile_widths[0] 
    
        ip_tile_y_min = ip_tile_y - (- delta_y + \
                        hsmlPart + tile_widths[1] / 2) // tile_widths[1] 
        ip_tile_y_max = ip_tile_y + (delta_y +   \
                        hsmlPart + tile_widths[1] / 2) // tile_widths[1] 
    
        ip_tile_z_min = ip_tile_z - (- delta_z + \
                        hsmlPart + tile_widths[2] / 2) // tile_widths[2] 
        ip_tile_z_max = ip_tile_z + (delta_z +   \
                        hsmlPart + tile_widths[2] / 2) // tile_widths[2]
        
        # filter_type == 0 is nearest-grid-point (NGP)
        if filter_type == 0:
            # these are the indices of the closest grid-point
            # if particle is in tile i, its xcoord is \in [x_i, x_(i+1))
            # if delta_x > 0 it means that particle is closer to
            # x_(i+1) than to x_i
            xgrid_idx = ip_tile_x
            if (delta_x > 0): xgrid_idx += 1
            ygrid_idx = ip_tile_y
            if (delta_y > 0): ygrid_idx += 1
            zgrid_idx = ip_tile_z
            if (delta_z > 0): zgrid_idx += 1

            # there could be out-of-bound problems
            # need to select only those that are on the grid
            if (xgrid_idx >= 0) and (xgrid_idx < nx):
                if (ygrid_idx >= 0) and (ygrid_idx < ny):
                    if (zgrid_idx >= 0) and (zgrid_idx < nz):

                        weight_tmp = weights[ip]
                        # the following two need to be atomic add
                        cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx), weight_tmp)
                        cuda.atomic.add(deposited_var, (xgrid_idx, ygrid_idx, zgrid_idx), variable[ip] * weight_tmp)

            
            
        # filter_type == 1 is cloud-in-cell (CIC), particle is a cube
        elif filter_type == 1:
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

                        # there could be out-of-bound problems
                        # need to select only those that are on the grid
                        if (xgrid_idx >= 0) and (xgrid_idx < nx):
                            if (ygrid_idx >= 0) and (ygrid_idx < ny):
                                if (zgrid_idx >= 0) and (zgrid_idx < nz):

                                    # write CIC deposition routine
                                    # weight in x dimension
                                    weight_tmp = weight_cic(xgrid[xgrid_idx], xp, hsmlPart, widths[0]) * \
                                    # weight = weight_tmp
                                    # weight in y dimension
                                    weight_tmp = 
                                    weight_cic(ygrid[ygrid_idx], yp, hsmlPart, widths[1]) * \
                                    weight *= weight_tmp
                                    # weight in z dimension
                                    weight_tmp = 
                                    weight_cic(zgrid[zgrid_idx], zp, hsmlPart, widths[2])
                                    weight *= weight_tmp
                                    # the following two need to be atomic add
                                    cuda.atomic.add(scratch, (xgrid_idx, ygrid_idx, zgrid_idx), weight)
                                    cuda.atomic.add(deposited_var, (xgrid_idx, ygrid_idx, zgrid_idx), variable[ip] * weight)

                                    
        # if weight > 0.:
        #     smooth_var[ip] /= weight


# In[ ]:


class Lorenzo(tc.DepositCartesianGrid):
    def 


# In[2]:


depo = Lorenzo(snap, center, widths,npix=128, threadsperblock=256, regionType='cartesian')


# In[ ]:




