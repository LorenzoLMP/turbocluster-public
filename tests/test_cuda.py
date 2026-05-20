import cupy as cp
from numba import cuda
import numpy as np

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

pos = cp.random.random((1000000,3))
isParticleInDomain = cp.zeros(pos.shape[0])
hsml = cp.zeros(pos.shape[0])
widths = cp.array([1,1,1])
center = cp.array([0.5,0.5,0.5])


Np = pos.shape[0]
threadsperblock = 256
blocks_1d = (Np + (threadsperblock - 1)) // threadsperblock

if __name__ == "__main__":

    try:
        check_particle[blocks_1d, threadsperblock](pos, hsml, center, widths, isParticleInDomain)
        success = np.array_equal(cp.asnumpy(isParticleInDomain),np.ones(pos.shape[0]))

        if success:
            print('The test worked')
        else:
            raise ValueError('The test failed.')
    except Exception as e:
        print(e)

