#-------------------------------------------------------------------------------
# This function defines the sea level change timeseries
# *Default = sinusoidal tidal cycle if 'tides' with 1m amplitude
#-------------------------------------------------------------------------------

import numpy as np
from params import t_final,nt_per_year

def sl_change(t):
    SLC = np.sin(4*np.pi*t/(3.154e7/12.0/30.0))  # tidal frequency of 2 per day
    return SLC
