#!/usr/bin/env python
# coding: utf-8

# In[1]:


import paicos as pa
import numpy as np
import cupy as cp
import turbocluster as tc
import math
from numba import cuda
import nvtx
import finufft
from IPython.display import Latex
import scipy.signal

pa.settings.strict_units = False
# A snapshot object
# snap = pa.Snapshot(pa.data_dir, 247)
# snap = pa.Snapshot('/lustre/astro/berlok/zoom-simulations-new-ics/halo_0003/adiabatic-mhd/zoom4_ics_v1/output', 247)
# snap = pa.Snapshot('/lustre/astro/berlok/zoom-simulations-new-ics/halo_0003/tng/zoom12_ics_v1/output', 247)
# snap = pa.Snapshot('/scratch/lperrone/zoom-simulations-new-ics/halo_0003/tng/zoom12_ics_v1/output', 247)
# snap = pa.Snapshot('/scratch/lperrone/zoom12_ics_v1/output', 247, basename='snap')
# snap = pa.Snapshot('/llust21/cosmo-plasm/zoom-simulations-arepo2/halo_0003/tng/zoom12/output',
#                    305, basename='snapshot')
snap = pa.Snapshot('/llust21/cosmo-plasm/zoom-simulations-arepo2/halo_0003/tng/zoom8/output', 
                   305, basename='snapshot')
center = snap.Cat.Group['GroupPos'][0]
widths = np.array([5e2, 5e2, 5e2], dtype=float) ## good for testing
# widths = np.array([2e1, 2e1, 2e1], dtype=float)

test_type = 'diff_of_gaussians'


# In[2]:


import cmasher as cmr
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

grays = plt.cm.gray(np.linspace(0,1,10))
reds = plt.cm.Reds_r(np.linspace(0,1,10))
blues = plt.cm.Blues_r(np.linspace(0,1,10))
greens = plt.cm.Greens_r(np.linspace(0,1,10))
oranges = plt.cm.Oranges_r(np.linspace(0,1,10))

# plt.rc('font', family='serif')
# plt.rcParams['font.serif'] = 'ITC Bookman'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize']= 16
plt.rcParams['ytick.labelsize']= 16
plt.rcParams['xtick.minor.size']= 4.0
plt.rcParams['ytick.minor.size']= 4.0
plt.rcParams['xtick.major.size']= 5.0
plt.rcParams['ytick.major.size']= 5.0
plt.rcParams['xtick.minor.width']= 0.7
plt.rcParams['ytick.minor.width']= 0.7
plt.rcParams['xtick.major.width']= 1.
plt.rcParams['ytick.major.width']= 1.

plt.rcParams['xtick.direction']= 'in'
plt.rcParams['ytick.direction']= 'in'


# In[3]:


num_cuda = cp.cuda.runtime.getDeviceCount()
devices = []
for i in range(0, num_cuda):
    devices += [f'cuda:{i}']


# In[4]:


devices


# In[5]:


def enforce_hermitian(amplitude_matrix):
    ## this is for odd matrices
    amplitudes = amplitude_matrix.copy()
    Nx, Ny, Nz = amplitudes.shape
    for kk in range(0,Nz//2):
        amplitudes[:,:,kk]=np.conj(np.flip(amplitudes)[:,:,kk])
    
    amplitudes[:Nx//2+1,:Ny//2+1,Nz//2]=np.conj(np.flip(amplitudes)[:Nx//2+1,:Ny//2+1,Nz//2])
    amplitudes[:Nx//2+1,Ny//2+1:,Nz//2]=np.conj(np.flip(amplitudes)[:Nx//2+1,Ny//2+1:,Nz//2])
    amplitudes[Nx//2+1:,Ny//2+1:,Nz//2]=np.conj(np.flip(amplitudes)[Nx//2+1:,Ny//2+1:,Nz//2])
    amplitudes[Nx//2+1:,:Ny//2+1,Nz//2]=np.conj(np.flip(amplitudes)[Nx//2+1:,:Ny//2+1,Nz//2])

    return amplitudes


# In[6]:


# cp.cuda.Device(3).use()
# with cp.cuda.Device(3):

widths_slicer = widths.copy()
widths_slicer[2] = 0.
center_slicer = center.copy
slicer = pa.Slicer(snap, center_slicer, widths_slicer, 'z', npix=512)
extent = slicer.centered_extent.to('Mpc')

# Gauss = snap.uq('G')
arepo_length = snap['0_Diameters'].uq
density_unit = snap['0_Density'].uq
kpc = snap.uq('kpc')

weight = '0_Volume'

filter_lengths_vec = [120, 80, 40, 20]

filter_length_max = filter_lengths_vec[0]*np.ones(snap['0_Diameters'].shape)*arepo_length

sf = tc.SmoothingFilter(snap, center, widths, npix=256, orientation=None, 
                        search_radius=6.1*filter_length_max.value)
# sf = tc.SmoothingFilter(snap, center, widths, npix=256, orientation=None, 
#                         search_radius=filter_length_max.value)

depo = tc.DepositCartesianGrid(snap, center, widths, npoints=256, 
                                   threadsperblock=256, 
                                   regionType='cartesian', kernel_type="TSC")


# In[7]:


## generate Kolmogorov IC
nmax = 257
nfreq_tot = nmax**3
rng = np.random.default_rng(seed=456789)

x = snap['0_Coordinates'][sf.index][:, 0].value - center[0].value
y = snap['0_Coordinates'][sf.index][:, 1].value - center[1].value
z = snap['0_Coordinates'][sf.index][:, 2].value - center[2].value

kx_vec = np.arange(-(nmax-1)//2,(nmax-1)//2+1)*(2.0*np.pi/widths[0])
ky_vec = np.arange(-(nmax-1)//2,(nmax-1)//2+1)*(2.0*np.pi/widths[1])
kz_vec = np.arange(-(nmax-1)//2,(nmax-1)//2+1)*(2.0*np.pi/widths[2])

KX, KY, KZ = np.meshgrid(kx_vec, ky_vec, kz_vec, sparse=False, indexing='ij')

K2 = KX**2 + KY**2 + KZ**2 
phases_rho = 2.0*np.pi*rng.uniform(low=-1.0, high=1.0, size=(nmax,nmax,nmax))

K2min = (2.0*np.pi/widths[0])**2 + (2.0*np.pi/widths[1])**2 + (2.0*np.pi/widths[2])**2
# K2min *=9
power_law_exponent = -5./3.
# energy per 3D mode \times k^2 = E(k) \sim k^power_law_exponent
# ==> energy per 3D mode \sim k^[(power_law_exponent-2)/2]
ampl = np.zeros(K2.shape)
ampl = np.where(K2>0,np.sqrt(K2/K2min)**(0.5*(power_law_exponent-2.)),0.0)
# ampl = np.where(K2==0,10.0,ampl)

amplitudes_rho = nfreq_tot*ampl*np.exp(1j*phases_rho)/(widths[0]*widths[1]*widths[2])        
amplitudes_rho = enforce_hermitian(amplitudes_rho)

synthetic_rhofield = np.zeros(snap['0_Density'].shape)
## redo this with finufft 
synthetic_rhofield[sf.index] = np.real(finufft.nufft3d2(((2.0*np.pi/widths[0])*x)%(2.*np.pi), 
                                                        ((2.0*np.pi/widths[1])*y)%(2.*np.pi), 
                                                        ((2.0*np.pi/widths[2])*z)%(2.*np.pi), 
                             amplitudes_rho, eps=1e-10, isign=-1))
# synthetic_rhofield[sf.index] = np.real(finufft.nufft3d2(((2.0*np.pi/widths[0])*x), 
#                                                         ((2.0*np.pi/widths[1])*y), 
#                                                         ((2.0*np.pi/widths[2])*z), 
#                              amplitudes_rho, eps=1e-10, isign=-1))

snap['0_synthetic_rhofield'] = synthetic_rhofield * density_unit


# In[8]:


deposited_var_orig = depo.deposit_variable('0_synthetic_rhofield', weight='0_Volume')
# if (var_type=="snap"):
#     powerspectr, k1d, _ = depo.power_spectrum1d(deposited_var,
#                                             window=scipy.signal.windows.hann)
# elif (var_type=="synthetic"):


# In[9]:


powerspectr_orig, k1d_orig, _ = depo.power_spectrum1d(deposited_var_orig)


# In[10]:


def chooseIC_and_filter_and_deposit(var_type="snap", filter_type="gaussian"):
    
    sliced_var_container = np.zeros((len(filter_lengths_vec),512,512))
    spectra_var_container = []
    
    if (var_type=="snap"):
        var_str = "0_Density"
    elif (var_type=="synthetic"):
        var_str = '0_synthetic_rhofield'
    else:
        print("error: your var_type must either be 'snap' or 'synthetic'")
        return sliced_var_container

    for i in range(len(filter_lengths_vec)):

        filter_length = filter_lengths_vec[i]*np.ones(snap['0_Diameters'].shape)*arepo_length
        rho_filtered, rho_remaining = tc.extract_turbulent_scalar(snap, sf, 
                                                  var_str, 6.*filter_length, 
                                                  weight, filter_type, iterative=False)
    
        sliced_var_container[i,:,:] = slicer.slice_variable(rho_filtered).value
        deposited_var = depo.deposit_variable(rho_filtered, weight='0_Volume')
        if (var_type=="snap"):
            powerspectr, k1d, _ = depo.power_spectrum1d(deposited_var,
                                                    window=scipy.signal.windows.hann)
        elif (var_type=="synthetic"):
            powerspectr, k1d, _ = depo.power_spectrum1d(deposited_var)

        spectra_var_container.append((powerspectr, k1d))
        
    
    return sliced_var_container, spectra_var_container
        


# In[11]:


def make_test_and_plot(sliced_var_container, spectra_var_container, 
                       var_type="snap", filter_type="gaussian"):

    fig, ax = plt.subplots(2,2,figsize=(10,10), sharex=True, sharey=True)
    
    for n in range(len(filter_lengths_vec)):
        i = n//2
        j = n%2

        effective_lambda = np.sqrt(2.0)*np.pi*filter_lengths_vec[n]
    
        vmax = np.max(np.abs(sliced_var_container[n,:,:]))
        vmin = np.max([np.min(np.abs(sliced_var_container[n,:,:])), 1e-2*vmax])
        norm = LogNorm(vmin=vmin,vmax=vmax)
        # cmap = cmr.fall
        cmap = cmr.eclipse
        if (var_type == "synthetic"):
            vmin = -vmax
            norm = Normalize(vmin=vmin,vmax=vmax)
            cmap = cmr.fusion
        
    
        im = ax[i,j].imshow(
            sliced_var_container[n,:,:], origin='lower', norm=norm, cmap=cmap, extent=extent.value)

        ax[i,j].plot([-0.2,-0.2+filter_lengths_vec[n]/1e3],[-0.2,-0.2],color='w',lw=2.)
        # ax[i,j].plot([-0.2,-0.2+effective_lambda/1e3],[-0.17,-0.17],color='k',lw=2.)
        ax[i,j].set_xlabel('x')
        ax[i,j].set_ylabel('z')
        ax[i,j].set_title('l=%.1f'%(filter_lengths_vec[n]))
    
        cbar = fig.colorbar(im, orientation='horizontal', shrink=0.7)
        cbar.set_label(r'$\rho$')
    
    fig.suptitle('Filter %s for var %s'%(filter_type, var_type),fontsize=18)
    
    fig.subplots_adjust(top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.15,
    wspace=0.0)
    
    plt.savefig('./../../plots/narrow-pass-filter/%s/%s-%s.pdf'%(test_type, filter_type, var_type), bbox_inches='tight', dpi=400)


    ## now spectra plots
    fig, ax = plt.subplots(figsize=(8,6))

    tt = np.logspace(np.log10(2e-2), 0, 100)

    ax.plot(k1d_orig.value, powerspectr_orig.value, ls='-', color=grays[6],
                markerfacecolor='none', label=r'original')
    for n in range(len(filter_lengths_vec)):
        powerspectr, k1d = spectra_var_container[n]
    
        ax.plot(k1d.value, powerspectr.value, ls='-', color=reds[2*n],
                markerfacecolor='none', label=r'l=%.1f'%(filter_lengths_vec[n]))
    
    ax.plot(tt, 1e9*tt**(-5./3.), ls='-', color='k',lw=1., label=r'$k^{-5/3}$')
    
    ax.set_xlabel('$k$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.legend(loc='best',ncols=1,fontsize=14)
    
    ax.set_title(r'Energy Spectral Density %s for var %s'%(filter_type, var_type), fontsize=18)
    
    # ax.set_xlim(xmin=1e-2)
    # ax.set_ylim(ymin=1e-25,ymax=1e-11)
    # ax.set_ylim(ymin=1e-20)
    # ax[1].set_xlim(xmax=30)
    savename = "power_spectrum_%s-%s"%(filter_type, var_type)
    
    plt.savefig('./../../plots/narrow-pass-filter/%s/'%(test_type)+savename+'.pdf',dpi=400)


    
    fig, ax = plt.subplots(figsize=(8,8), sharex=True, sharey=True)
    
    vmax = np.max(np.abs(deposited_var_orig.value))
    vmin = -vmax
    norm = Normalize(vmin=vmin,vmax=vmax)
    cmap = cmr.fusion
        
    
    im = ax.imshow(
        deposited_var_orig[:,:,0].value, origin='lower', norm=norm, cmap=cmap, extent=extent.value)
    
    # ax[i,j].plot([-0.2,-0.2+filter_lengths_vec[n]/1e3],[-0.2,-0.2],color='w',lw=2.)
    # ax[i,j].plot([-0.2,-0.2+effective_lambda/1e3],[-0.17,-0.17],color='k',lw=2.)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    # ax[i,j].set_title('l=%.1f'%(filter_lengths_vec[n]))
    
    cbar = fig.colorbar(im, orientation='horizontal', shrink=0.7)
    cbar.set_label(r'$\rho$')
    
    fig.suptitle('Original var %s'%(var_type),fontsize=18)
    
    fig.subplots_adjust(top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.15,
    wspace=0.0)
    
    savename = "original_var-%s"%(var_type)
    plt.savefig('./../../plots/narrow-pass-filter/%s/'%(test_type)+savename+'.pdf',dpi=400)


    
    plt.show()
    


# In[12]:


# for synthetic Kolmogorov spectrum with mexican filter
sliced_var_synthetic_mexican, spectra_var_synthetic_mexican = chooseIC_and_filter_and_deposit(var_type="synthetic", 
                                                                                              filter_type='mexican-hat')
make_test_and_plot(sliced_var_synthetic_mexican, spectra_var_synthetic_mexican, 
                   var_type="synthetic",filter_type='mexican-hat')


# In[13]:


# for synthetic Kolmogorov spectrum with gaussian filter
sliced_var_synthetic_gaussian, spectra_var_synthetic_gaussian = chooseIC_and_filter_and_deposit(var_type="synthetic", 
                                                                                    filter_type='gaussian')
make_test_and_plot(sliced_var_synthetic_gaussian, spectra_var_synthetic_gaussian, 
                   var_type="synthetic",filter_type='gaussian')


# In[ ]:




