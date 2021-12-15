#-------------------------------------------------------------------------------
# This function defines the sea level change timeseries for marine ice sheet problem.
# *Default = sinusoidal tidal cycle if 'tides' with 1m amplitude if 'tides' turned 'on', OR...
#          = zero if 'tides' turned 'off'
#-------------------------------------------------------------------------------

import numpy as np
from params import t_final,nt_per_year,tides

def sl_change(t):
    if tides == 'on':
        SLC = np.sin(4*np.pi*t/(3.154e7/12.0/30.0))  # tidal frequency of 2 per day
    else:
        SLC = 0.0                                    # no sea level change for
                                                     # long-time marine problem
    return SLC
