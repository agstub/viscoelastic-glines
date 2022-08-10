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
    Bed = -np.abs(600*x/(0.5*Lngth) - 600)
    return Bed

#------------------Generate initial ice-water/ice-bed interface-----------------
def interface(x):
    Int = 0.5 * bed(x + 0.5*Lngth*bed(x)/600)
    return Int
#-------------------------------------------------------------------------------

#
# import matplotlib.pyplot as plt
# from params import X_fine
# plt.plot(X_fine,bed(X_fine))
# plt.plot(X_fine,interface(X_fine))
# plt.show()
