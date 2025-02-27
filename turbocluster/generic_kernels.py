import cupy as cp
from numba import cuda
import math

epsilon = 1e-2 

@cuda.jit(device=True, inline=True)
def distance(pos, pos_other):
    dist = math.sqrt((pos[0] - pos_other[0])**2 +
                     (pos[1] - pos_other[1])**2 +
                     (pos[2] - pos_other[2])**2)
    return dist

@cuda.jit(device=True, inline=True)
def sphere_kernel(dist, filter_length):

    # weight = math.exp(-0.5*(dist/filter_length)**2)
    weight = 1./(4.*cp.pi*filter_length**3/3.0)

    return weight

@cuda.jit(device=True, inline=True)
def gaussian_kernel(dist, filter_length):

    # weight = math.exp(-0.5*(dist/filter_length)**2)
    weight = math.exp(-0.5*(dist/filter_length)**2)/filter_length**3/(2.0*cp.pi)**(3./2.)

    return weight

@cuda.jit(device=True, inline=True)
def mexican_kernel(dist, filter_length):

    weight = (3 - (dist/filter_length)**2)
    weight *= math.exp(-0.5*(dist/filter_length)**2)/filter_length**3/(2.0*cp.pi)**(3./2.)

    return weight

@cuda.jit(device=True, inline=True)
def mexican_kernel_2(dist, filter_length):

    weight_1 = gaussian_kernel(dist, filter_length/(1. + epsilon)**0.5 )
    weight_2 = gaussian_kernel(dist, filter_length*(1. + epsilon)**0.5 )
    weight = (weight_1 - weight_2)/epsilon

    return weight

@cuda.jit(device=True, inline=True)
def check_distance(ip_tile_x, ip_tile_y, ip_tile_z,
                   tile_x, tile_y, tile_z,
                   delta_x, delta_y, delta_z,
                   tile_widths, filter_length):

    overlap = False

    xcoord_edge = delta_x
    if (tile_x > ip_tile_x):
        xcoord_edge = tile_widths[0] * (tile_x - ip_tile_x - 0.5)
    elif (tile_x < ip_tile_x):
        xcoord_edge = tile_widths[0] * (tile_x - ip_tile_x + 0.5)

    ycoord_edge = delta_y
    if (tile_y > ip_tile_y):
        ycoord_edge = tile_widths[1] * (tile_y - ip_tile_y - 0.5)
    elif (tile_y < ip_tile_y):
        ycoord_edge = tile_widths[1] * (tile_y - ip_tile_y + 0.5)

    zcoord_edge = delta_z
    if (tile_z > ip_tile_z):
        zcoord_edge = tile_widths[2] * (tile_z - ip_tile_z - 0.5)
    elif (tile_z < ip_tile_z):
        zcoord_edge = tile_widths[2] * (tile_z - ip_tile_z + 0.5)
        
    dist2 = (delta_x - xcoord_edge)**2 + \
            (delta_y - ycoord_edge)**2 + \
            (delta_z - zcoord_edge)**2

    filt2 = filter_length**2
    if (filt2 >= dist2):
        overlap = True
    
    return overlap


@cuda.jit()
def check_particle(pos, hsml, center, widths, isParticleInDomain):
    """
    """
    ip = cuda.grid(1)
    # each thread is assigned to a particle
    numParticles = pos.shape[0]
    isParticleInDomainTmp = 0

    if (ip < numParticles):

        xp, yp, zp = pos[ip]
        xmin = center[0] - widths[0] / 2 - hsml[ip]
        xmax = center[0] + widths[0] / 2 + hsml[ip]

        ymin = center[1] - widths[1] / 2 - hsml[ip]
        ymax = center[1] + widths[1] / 2 + hsml[ip]

        zmin = center[2] - widths[2] / 2 - hsml[ip]
        zmax = center[2] + widths[2] / 2 + hsml[ip]

        if (xp > xmin) and (xp < xmax):
            if (yp > ymin) and (yp < ymax):
                if (zp > zmin) and (zp < zmax):
                    isParticleInDomainTmp = 1


        isParticleInDomain[ip] = isParticleInDomainTmp

@cuda.jit()
def compactify_particles(pos, tile_index, cumulative_occupancy_flat, isParticleInDomain,
                         oldIndex):
    """
    """
    ip = cuda.grid(1)
    numParticles = pos.shape[0]
    # each thread takes care of a particle
    if (ip < numParticles):
        newPos = int(cumulative_occupancy_flat[ip])
        if (isParticleInDomain[ip] > 0):
            oldIndex[newPos - 1] = ip

