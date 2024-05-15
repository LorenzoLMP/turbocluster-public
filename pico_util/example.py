import paicos as pa
import numpy as np
pa.settings.strict_units = False
pa.print_info_when_deriving_variables(False)

import pico
halonums = pico.halonums

simfolder = pico.get_simfolder(104)
info = pico.PicoOutputInfo(f'{simfolder}output/pico_output_info.txt')
tracking = np.loadtxt(f'{simfolder}output/pico_tracking_info.txt', dtype=int)

info.num_vec_full
