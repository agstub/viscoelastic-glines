#-------------------------------------------------------------------------------
# Define bed topography and initial ice-water interface functions.
# They should be equal on one margin of the domain for the
# marine ice sheet problem (i.e., the grounded portion).
# NOTE: initial ice-water elevation should be set to z=0 (i.e. interface=0 where interface>bed)
#-------------------------------------------------------------------------------

import numpy as np
from params import Lngth,Hght

#-------------------- Generate Bed Topography-----------------------------------
def bed(x):
    Bed =  -2*np.abs(x/(0.5*Lngth) - 1)
    return Bed

#------------------Generate initial ice-water/ice-bed interface-----------------
def interface(x):
    Int = 0.5*bed(x)*(1-np.sign(x-0.5*Lngth))
    return Int
#-------------------------------------------------------------------------------
