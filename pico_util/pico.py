import paicos as pa
import numpy as np
pa.settings.strict_units = False
# pa.print_info_when_deriving_variables(False)

from pico_output_info import PicoOutputInfo

def get_folder(basefolder, halo, galform, zoom_factor, descr):
    # specials = [417, 20, 542]  # Special cases
    # if halo in specials:
    #     descr += '_special'
    zoom = zoom_factor
    if galform is True:
        phys = 'gal-form'
    elif galform == 'tng':
        phys = 'tng'
    elif galform == 'dark':
        phys = 'dark'
    else:
        phys = 'adiabatic-mhd'
        
    if basefolder[-1] != '/':
        basefolder += '/'
    
    return basefolder + f"halo_{halo:04d}/{phys}/zoom{zoom}{descr}/"

def get_simfolder(halo, galform='tng', zoom_factor=12, descr=''):
    folder = get_folder(base_sim_folder, halo=halo, galform=galform, zoom_factor=zoom_factor, descr=descr)
    return folder

def get_analysisfolder(halo, galform='tng', zoom_factor=12, descr=''):
    folder = get_folder(base_analysis_folder, halo=halo, galform=galform, zoom_factor=zoom_factor, descr=descr)
    return folder

base_analysis_folder = '/llust21/cosmo-plasm/analysis-zoom-simulations-arepo2'
base_sim_folder = '/llust21/cosmo-plasm/zoom-simulations-arepo2'

# All 25 halo numbers that we want to do
halonums = [0, 1, 2, 3, 4, 8, 9, 19, 20, 22, 33, 41, 49, 56, 98, 104, 112, 194, 239, 260, 261, 344, 417, 420, 542]

# Halonums that are done and transferred
all_halonums = [0, 1, 2, 3, 4, 8, 9, 19, 20, 22, 33, 41, 49, 56, 98, 104, 112, 194, 239, 260, 261, 344, 417, 420, 542]

halonums_single = [1, 22, 112, 239, 417, 542]

halonums_multi = []
for halo in all_halonums:
    if halo not in halonums_single:
        halonums_multi.append(halo)

halonums = list(all_halonums)

still_running = []
transferring = []

for halo in still_running + transferring:
    halonums.remove(halo)

# halonums = [1]